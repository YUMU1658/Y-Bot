"""Bot 核心类。

框架的核心协调者，负责初始化各模块、启动 WebSocket 服务端、
处理事件并格式化输出日志。
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any

from ybot import __version__
from ybot.core.config import Config
from ybot.core.ws_server import WebSocketServer
from ybot.models.event import (
    Event,
    GroupMessageEvent,
    MessageEvent,
    MetaEvent,
    NoticeEvent,
    PrivateMessageEvent,
    RequestEvent,
    parse_event,
)
from ybot.models.message import segments_to_text
from ybot.services.ai_chat import AIChatService
from ybot.services.bot_info import BotInfoService
from ybot.services.env_builder import EnvBuilder, MessageFormatter
from ybot.storage.conversation import ConversationStore
from ybot.utils.logger import get_logger, setup_logger


class Bot:
    """Y-BOT 核心类。

    负责协调配置、WebSocket 服务端和事件处理。

    Attributes:
        config: 全局配置。
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # 初始化主 Logger
        self._logger = setup_logger("Y-BOT", config.bot.log_level)
        self._msg_logger = get_logger("消息")
        self._meta_logger = get_logger("元事件")
        self._notice_logger = get_logger("通知")
        self._request_logger = get_logger("请求")

        # 初始化 WebSocket 服务端
        self._ws_server = WebSocketServer(
            host=config.server.host,
            port=config.server.port,
        )
        self._ws_server.set_event_handler(self._on_raw_event)

        # 初始化 AI 对话服务
        self._conv_store = ConversationStore()
        self._ai_chat = AIChatService(config.ai, self._conv_store)

        # 初始化 Bot 信息缓存、ENV 构建器、消息格式化器
        self._bot_info = BotInfoService(self._ws_server)
        self._env_builder = EnvBuilder(self._bot_info)
        self._msg_formatter = MessageFormatter(self._bot_info)

        self._running = False
        self._bot_info_initialized = False

    def run(self) -> None:
        """启动 Bot，进入事件循环。

        此方法会阻塞直到收到退出信号（Ctrl+C）。
        """
        self._logger.info(f"Y-BOT v{__version__} 正在启动...")

        try:
            asyncio.run(self._async_run())
        except KeyboardInterrupt:
            pass  # 优雅退出，_async_run 中已处理清理

    async def _async_run(self) -> None:
        """异步主循环。"""
        self._running = True

        # 注册信号处理（仅 Unix）
        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._signal_handler)

        # 启动 AI 服务（需在事件循环中创建 aiohttp 会话）
        await self._conv_store.initialize()
        await self._ai_chat.start()
        await self._ws_server.start()

        self._logger.info(f"Y-BOT v{__version__} 已就绪，等待 OneBot 客户端连接...")

        # 保持运行直到收到退出信号
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    def _signal_handler(self) -> None:
        """处理退出信号。"""
        self._running = False

    async def shutdown(self) -> None:
        """优雅关闭 Bot。"""
        self._logger.info("正在关闭 Y-BOT...")
        await self._ws_server.stop()
        await self._ai_chat.stop()
        await self._conv_store.close()
        self._logger.info("Y-BOT 已关闭")

    async def _on_raw_event(self, data: dict[str, Any]) -> None:
        """原始事件数据回调，解析并分发事件。"""
        try:
            event = parse_event(data)
            await self._handle_event(event)
        except Exception as e:
            self._logger.error(f"处理事件时发生错误: {e}")

    async def _handle_event(self, event: Event) -> None:
        """事件处理入口。

        根据事件类型进行日志输出和 AI 回复触发。
        """
        # 过滤 Bot 自身发送的消息，防止自回复死循环
        if isinstance(event, MessageEvent) and event.user_id == event.self_id:
            return

        # 首次收到事件时，延迟初始化 BotInfoService（需要 OneBot 客户端已连接）
        if not self._bot_info_initialized:
            self._bot_info_initialized = True
            try:
                await self._bot_info.initialize()
            except Exception as e:
                self._logger.warning(
                    f"BotInfoService 初始化失败（将在后续请求中重试）: {e}"
                )

        match event:
            case GroupMessageEvent() as e:
                self._log_group_message(e)
                if self._is_at_me(e):
                    text = self._extract_text(e)
                    if text.strip():
                        session_key = f"group_{e.group_id}"
                        # 构建 ENV 头部
                        env_header = await self._env_builder.build_group_env(e.group_id)
                        # 格式化消息（附带发送者元信息）
                        formatted_msg = await self._msg_formatter.format_group_message(
                            e, text
                        )
                        reply = await self._ai_chat.chat(
                            session_key, formatted_msg, env_header
                        )
                        await self.send_group_msg(e.group_id, reply)
            case PrivateMessageEvent() as e:
                self._log_private_message(e)
                text = self._extract_text(e)
                if text.strip():
                    # 临时会话：sub_type 为 "group" 表示从群聊发起的临时私聊
                    if e.sub_type == "group":
                        temp_group_id = e.raw_data.get("sender", {}).get("group_id", 0)
                        session_key = f"temp_{temp_group_id}_{e.user_id}"
                        env_header = await self._env_builder.build_temp_env(
                            e.user_id, temp_group_id
                        )
                    else:
                        session_key = f"friend_{e.user_id}"
                        env_header = await self._env_builder.build_private_env(
                            e.user_id
                        )
                    # 格式化消息
                    formatted_msg = self._msg_formatter.format_private_message(e, text)
                    reply = await self._ai_chat.chat(
                        session_key, formatted_msg, env_header
                    )
                    await self.send_private_msg(e.user_id, reply)
            case MessageEvent() as e:
                # 兜底：未知消息类型
                text = segments_to_text(e.message)
                self._msg_logger.info(f"[{e.message_type}] 用户:{e.user_id} | {text}")
            case MetaEvent() as e:
                self._log_meta_event(e)
            case NoticeEvent() as e:
                self._log_notice_event(e)
            case RequestEvent() as e:
                self._log_request_event(e)
            case _:
                self._logger.debug(f"收到未知事件类型: {event.post_type}")

    # ---- 消息发送 ----

    async def send_group_msg(self, group_id: int, message: str) -> None:
        """发送群聊消息。

        Args:
            group_id: 目标群号。
            message: 消息文本。
        """
        await self._ws_server.send_api(
            "send_group_msg",
            {
                "group_id": group_id,
                "message": message,
            },
        )

    async def send_private_msg(self, user_id: int, message: str) -> None:
        """发送私聊消息。

        Args:
            user_id: 目标用户 QQ 号。
            message: 消息文本。
        """
        await self._ws_server.send_api(
            "send_private_msg",
            {
                "user_id": user_id,
                "message": message,
            },
        )

    # ---- 辅助方法 ----

    @staticmethod
    def _is_at_me(event: GroupMessageEvent) -> bool:
        """检查群聊消息是否 @了机器人。

        Args:
            event: 群聊消息事件。

        Returns:
            如果消息中包含 @机器人 则返回 True。
        """
        bot_qq = str(event.self_id)
        for seg in event.message:
            if seg.type == "at" and seg.data.get("qq") == bot_qq:
                return True
        return False

    @staticmethod
    def _extract_text(event: MessageEvent) -> str:
        """从消息段中提取纯文本内容。

        过滤掉 at 段和其他非文本段，仅保留文本内容并去除首尾空白。

        Args:
            event: 消息事件。

        Returns:
            提取的纯文本。
        """
        parts: list[str] = []
        for seg in event.message:
            if seg.type == "text":
                parts.append(seg.data.get("text", ""))
        return "".join(parts).strip()

    def _log_group_message(self, event: GroupMessageEvent) -> None:
        """格式化输出群聊消息日志。"""
        text = segments_to_text(event.message)
        nickname = event.sender.card or event.sender.nickname or str(event.user_id)
        self._msg_logger.info(
            f"[群聊] 群:{event.group_id} | 用户:{event.user_id}({nickname}) | {text}"
        )

    def _log_private_message(self, event: PrivateMessageEvent) -> None:
        """格式化输出私聊消息日志。"""
        text = segments_to_text(event.message)
        nickname = event.sender.nickname or str(event.user_id)
        self._msg_logger.info(f"[私聊] 用户:{event.user_id}({nickname}) | {text}")

    def _log_meta_event(self, event: MetaEvent) -> None:
        """记录元事件日志。"""
        match event.meta_event_type:
            case "heartbeat":
                self._meta_logger.debug(f"心跳 | self_id:{event.self_id}")
            case "lifecycle":
                sub_type = event.raw_data.get("sub_type", "unknown")
                self._meta_logger.info(
                    f"生命周期 | {sub_type} | self_id:{event.self_id}"
                )
            case _:
                self._meta_logger.debug(
                    f"{event.meta_event_type} | self_id:{event.self_id}"
                )

    def _log_notice_event(self, event: NoticeEvent) -> None:
        """记录通知事件日志。"""
        extra_info = ""
        if "group_id" in event.raw_data:
            extra_info = f" | 群:{event.raw_data['group_id']}"
        if "user_id" in event.raw_data:
            extra_info += f" | 用户:{event.raw_data['user_id']}"

        self._notice_logger.info(f"{event.notice_type}{extra_info}")

    def _log_request_event(self, event: RequestEvent) -> None:
        """记录请求事件日志。"""
        extra_info = ""
        if "user_id" in event.raw_data:
            extra_info = f" | 用户:{event.raw_data['user_id']}"
        if "group_id" in event.raw_data:
            extra_info += f" | 群:{event.raw_data['group_id']}"

        self._request_logger.info(f"{event.request_type}{extra_info}")
