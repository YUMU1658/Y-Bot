"""Bot 核心类。"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any

from ybot import __version__
from ybot.commands import CommandRegistry
from ybot.commands.clear import ClearCommand
from ybot.core.config import Config
from ybot.core.event_handler import EventHandler
from ybot.core.request_queue import RequestQueue
from ybot.core.ws_server import WebSocketServer
from ybot.services.ai_chat import AIChatService
from ybot.services.ai_request_handler import AIRequestHandler
from ybot.services.bot_info import BotInfoService
from ybot.services.command_handler import CommandHandler
from ybot.services.env_builder import EnvBuilder, MessageFormatter
from ybot.services.event_logger import EventLogger
from ybot.services.interceptor import InterceptorService
from ybot.services.llm_client import LLMClient
from ybot.services.message_content import MessageContentResolver
from ybot.services.message_context import MessageContextCollector
from ybot.services.message_sender import MessageSender
from ybot.services.notice_handler import NoticeHandler
from ybot.services.poke_limiter import PokeLimiter
from ybot.services.worldbook import WorldBookService
from ybot.storage.chat_log import SessionChatLog
from ybot.storage.conversation import ConversationStore
from ybot.tools import ToolRegistry
from ybot.tools.contact_info import ContactInfoTool
from ybot.tools.group_info import GroupInfoTool
from ybot.tools.recall_msg import RecallMsgTool
from ybot.tools.viewer import ViewerTool
from ybot.utils.logger import setup_logger


class Bot:
    """Y-BOT 核心类，负责依赖装配、生命周期和发送兼容门面。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._logger = setup_logger("Y-BOT", config.bot.log_level)

        self._ws_server = WebSocketServer(
            host=config.server.host,
            port=config.server.port,
            access_token=config.server.access_token,
        )

        self._conv_store = ConversationStore()

        self._worldbook: WorldBookService | None = None
        if config.worldbook.enabled:
            self._worldbook = WorldBookService(
                worldbook_dir=config.worldbook.worldbook_dir,
                enabled_books=config.worldbook.enabled_books,
                enabled=True,
            )
            self._worldbook.load()

        self._bot_info = BotInfoService(self._ws_server)
        self._env_builder = EnvBuilder(self._bot_info)
        self._msg_formatter = MessageFormatter(self._bot_info)
        self._chat_log = SessionChatLog(buffer_size=config.ai.context_buffer)

        self._poke_limiter = PokeLimiter(
            cooldown=config.poke.cooldown,
            daily_limit=config.poke.daily_limit,
        )
        self._msg_sender = MessageSender(
            ws_server=self._ws_server,
            bot_info=self._bot_info,
            chat_log=self._chat_log,
            poke_limiter=self._poke_limiter,
        )

        self._tool_registry: ToolRegistry | None = None
        if config.tools.enabled:
            self._tool_registry = ToolRegistry(
                ws_server=self._ws_server,
                bot_info=self._bot_info,
                chat_log=self._chat_log,
                enable_vision=config.ai.enable_vision,
            )
            self._tool_registry.register(RecallMsgTool())
            self._tool_registry.register(GroupInfoTool())
            self._tool_registry.register(ContactInfoTool())
            self._tool_registry.register(ViewerTool())

        self._llm_client = LLMClient()
        self._ai_chat = AIChatService(
            config.ai,
            self._conv_store,
            worldbook=self._worldbook,
            tool_registry=self._tool_registry,
            llm_client=self._llm_client,
        )

        self._request_queue = RequestQueue(debounce_seconds=1.0)

        self._interceptor: InterceptorService | None = None
        if config.interceptor.enabled:
            self._interceptor = InterceptorService(
                config.interceptor, config.ai, llm_client=self._llm_client
            )

        self._cmd_registry: CommandRegistry | None = None
        if config.commands.enabled:
            self._cmd_registry = CommandRegistry()
            self._cmd_registry.register(
                ClearCommand(self._conv_store, self._chat_log)
            )

        self._event_logger = EventLogger()
        self._content_resolver = MessageContentResolver(
            self._ws_server, self._chat_log
        )
        self._context_collector = MessageContextCollector(
            config.ai, self._chat_log, self._conv_store, self._content_resolver
        )
        self._command_handler = CommandHandler(
            self._cmd_registry, config.commands, self._ws_server
        )
        self._notice_handler = NoticeHandler(self._bot_info, self._chat_log)
        self._ai_request_handler = AIRequestHandler(
            config=config,
            ai_chat=self._ai_chat,
            message_sender=self._msg_sender,
            request_queue=self._request_queue,
            interceptor=self._interceptor,
        )
        if self._interceptor:
            self._request_queue.set_interrupt_callback(
                self._ai_request_handler.on_interrupt_check
            )

        self._event_handler = EventHandler(
            bot_info=self._bot_info,
            env_builder=self._env_builder,
            msg_formatter=self._msg_formatter,
            chat_log=self._chat_log,
            request_queue=self._request_queue,
            event_logger=self._event_logger,
            content_resolver=self._content_resolver,
            context_collector=self._context_collector,
            command_handler=self._command_handler,
            notice_handler=self._notice_handler,
            ai_request_handler=self._ai_request_handler,
        )
        self._ws_server.set_event_handler(self._event_handler.on_raw_event)

        self._running = False

    def run(self) -> None:
        """启动 Bot，进入事件循环。"""
        self._logger.info(f"Y-BOT v{__version__} 正在启动...")

        try:
            asyncio.run(self._async_run())
        except KeyboardInterrupt:
            pass

    async def _async_run(self) -> None:
        """异步主循环。"""
        self._running = True

        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._signal_handler)
        else:
            signal.signal(signal.SIGINT, self._win32_signal_handler)
            signal.signal(signal.SIGTERM, self._win32_signal_handler)

        await self._conv_store.initialize()
        await self._llm_client.start()
        await self._ai_chat.start()
        if self._interceptor:
            await self._interceptor.start()
        await self._request_queue.start()
        await self._ws_server.start()

        self._logger.info(f"Y-BOT v{__version__} 已就绪，等待 OneBot 客户端连接...")

        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    def _signal_handler(self) -> None:
        """处理退出信号（Unix）。"""
        self._running = False

    def _win32_signal_handler(self, signum: int, frame: Any) -> None:
        """处理退出信号（Windows）。"""
        self._running = False

    async def shutdown(self) -> None:
        """优雅关闭 Bot。"""
        self._logger.info("正在关闭 Y-BOT...")
        await self._ws_server.stop()
        await self._request_queue.stop()
        if self._interceptor:
            await self._interceptor.stop()
        await self._ai_chat.stop()
        await self._llm_client.stop()
        await self._conv_store.close()
        self._logger.info("Y-BOT 已关闭")

    async def send_group_msg(
        self,
        group_id: int,
        message: str,
        reply_id: int | None = None,
        *,
        session_key: str | None = None,
    ) -> int | None:
        """发送群聊消息（委托给 MessageSender）。"""
        return await self._msg_sender.send_group_msg(
            group_id, message, reply_id, session_key=session_key
        )

    async def send_private_msg(
        self,
        user_id: int,
        message: str,
        reply_id: int | None = None,
        *,
        session_key: str | None = None,
    ) -> int | None:
        """发送私聊消息（委托给 MessageSender）。"""
        return await self._msg_sender.send_private_msg(
            user_id, message, reply_id, session_key=session_key
        )

    async def send_poke(
        self, target_id: int, group_id: int | None = None
    ) -> tuple[bool, str]:
        """发送戳一戳（委托给 MessageSender）。"""
        return await self._msg_sender.send_poke(target_id, group_id)
