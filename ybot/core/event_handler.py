"""OneBot 事件分发处理器。"""

from __future__ import annotations

from typing import Any

from ybot.core.request_queue import QueuedMessage, RequestQueue
from ybot.models.event import (
    Event,
    GroupMessageEvent,
    MessageEvent,
    MetaEvent,
    NoticeEvent,
    PokeNoticeEvent,
    PrivateMessageEvent,
    RequestEvent,
    parse_event,
)
from ybot.models.message import segments_to_text
from ybot.services.ai_request_handler import AIRequestHandler
from ybot.services.bot_info import BotInfoService
from ybot.services.command_handler import CommandHandler
from ybot.services.env_builder import EnvBuilder, MessageFormatter
from ybot.services.event_logger import EventLogger
from ybot.services.message_content import MessageContentResolver
from ybot.services.message_context import MessageContextCollector
from ybot.services.notice_handler import NoticeHandler
from ybot.storage.chat_log import SessionChatLog
from ybot.utils.logger import get_logger


class EventHandler:
    """解析原始事件并按事件类型委托给对应服务。"""

    def __init__(
        self,
        *,
        bot_info: BotInfoService,
        env_builder: EnvBuilder,
        msg_formatter: MessageFormatter,
        chat_log: SessionChatLog,
        request_queue: RequestQueue,
        event_logger: EventLogger,
        content_resolver: MessageContentResolver,
        context_collector: MessageContextCollector,
        command_handler: CommandHandler,
        notice_handler: NoticeHandler,
        ai_request_handler: AIRequestHandler,
    ) -> None:
        self._bot_info = bot_info
        self._env_builder = env_builder
        self._msg_formatter = msg_formatter
        self._chat_log = chat_log
        self._request_queue = request_queue
        self._event_logger = event_logger
        self._content_resolver = content_resolver
        self._context_collector = context_collector
        self._command_handler = command_handler
        self._notice_handler = notice_handler
        self._ai_request_handler = ai_request_handler
        self._logger = get_logger("Y-BOT")
        self._msg_logger = get_logger("消息")
        self._bot_info_initialized = False

    async def on_raw_event(self, data: dict[str, Any]) -> None:
        """原始事件数据回调，解析并分发事件。"""
        try:
            event = parse_event(data)
            await self.handle_event(event)
        except Exception as e:
            self._logger.error(f"处理事件时发生错误: {e}")

    async def handle_event(self, event: Event) -> None:
        """事件处理入口。"""
        if isinstance(event, MessageEvent) and event.user_id == event.self_id:
            if isinstance(event, GroupMessageEvent):
                sk = f"group_{event.group_id}"
                if not self._chat_log.has_message(sk, event.message_id):
                    await self._context_collector.collect_message(
                        event, session_key=sk, is_bot=True
                    )
            elif isinstance(event, PrivateMessageEvent):
                sk = self._context_collector.resolve_bot_private_session_key(event)
                if sk and not self._chat_log.has_message(sk, event.message_id):
                    await self._context_collector.collect_message(
                        event, session_key=sk, is_bot=True
                    )
            return

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
                await self._handle_group_message(e)
            case PrivateMessageEvent() as e:
                await self._handle_private_message(e)
            case MessageEvent() as e:
                text = segments_to_text(e.message)
                self._msg_logger.info(f"[{e.message_type}] 用户:{e.user_id} | {text}")
            case MetaEvent() as e:
                self._event_logger.log_meta_event(e)
            case PokeNoticeEvent() as e:
                self._event_logger.log_notice_event(e)
                await self._notice_handler.handle_poke(e)
            case NoticeEvent() as e:
                self._event_logger.log_notice_event(e)
                await self._notice_handler.handle_recall(e)
            case RequestEvent() as e:
                self._event_logger.log_request_event(e)
            case _:
                self._logger.debug(f"收到未知事件类型: {event.post_type}")

    async def _handle_group_message(self, event: GroupMessageEvent) -> None:
        self._event_logger.log_group_message(event)
        session_key = f"group_{event.group_id}"
        await self._context_collector.collect_message(
            event, session_key=session_key, is_bot=False
        )

        is_at_bot = self._is_at_me(event)
        cmd_text = self._content_resolver.extract_command_text(event)
        cmd_result = await self._command_handler.try_handle(
            event, session_key, cmd_text, is_at_bot
        )
        if cmd_result is True:
            return

        if not is_at_bot:
            return

        text = self._content_resolver.extract_content(event)
        text = await self._content_resolver.resolve_reply(text, event)
        text = await self._content_resolver.resolve_forward(text, event)
        if not text.strip():
            return

        env_header = await self._env_builder.build_group_env(event.group_id)
        formatted_msg = await self._msg_formatter.format_group_message(event, text)
        context_msg, last_ref_id = await self._context_collector.build_context_user_message(
            session_key, event.message_id, formatted_msg
        )
        image_urls = self._content_resolver.extract_image_urls(event)
        group_info = await self._bot_info.get_group_info(event.group_id)
        display_name = group_info.group_name or f"群{event.group_id}"

        queued = QueuedMessage(
            formatted_msg=formatted_msg,
            context_data={
                "type": "group",
                "session_key": session_key,
                "env_header": env_header,
                "context_msg": context_msg,
                "last_ref_id": last_ref_id,
                "image_urls": image_urls,
                "group_id": event.group_id,
                "display_name": display_name,
            },
        )
        self._request_queue.submit(session_key, queued, self._ai_request_handler.process)

    async def _handle_private_message(self, event: PrivateMessageEvent) -> None:
        self._event_logger.log_private_message(event)
        text = self._content_resolver.extract_content(event)
        text = await self._content_resolver.resolve_reply(text, event)
        text = await self._content_resolver.resolve_forward(text, event)
        if not text.strip():
            return

        if event.sub_type == "group":
            temp_group_id = event.raw_data.get("sender", {}).get("group_id", 0)
            session_key = f"temp_{temp_group_id}_{event.user_id}"
        else:
            temp_group_id = 0
            session_key = f"friend_{event.user_id}"

        cmd_text = self._content_resolver.extract_command_text(event)
        cmd_result = await self._command_handler.try_handle(
            event, session_key, cmd_text, is_at_bot=True
        )
        if cmd_result is True:
            return

        if event.sub_type == "group":
            env_header = await self._env_builder.build_temp_env(
                event.user_id, temp_group_id, event.sender.nickname
            )
        else:
            env_header = await self._env_builder.build_private_env(
                event.user_id, event.sender.nickname
            )

        await self._context_collector.collect_message(
            event, session_key=session_key, is_bot=False
        )
        formatted_msg = self._msg_formatter.format_private_message(event, text)
        context_msg, last_ref_id = await self._context_collector.build_context_user_message(
            session_key, event.message_id, formatted_msg
        )
        image_urls = self._content_resolver.extract_image_urls(event)
        display_name = event.sender.nickname or str(event.user_id)
        if event.sub_type == "group":
            temp_group_info = await self._bot_info.get_group_info(temp_group_id)
            temp_group_name = temp_group_info.group_name or str(temp_group_id)
            display_name = f"{display_name} ← {temp_group_name}"

        queued = QueuedMessage(
            formatted_msg=formatted_msg,
            context_data={
                "type": "private",
                "session_key": session_key,
                "env_header": env_header,
                "context_msg": context_msg,
                "last_ref_id": last_ref_id,
                "image_urls": image_urls,
                "user_id": event.user_id,
                "display_name": display_name,
            },
        )
        self._request_queue.submit(session_key, queued, self._ai_request_handler.process)

    @staticmethod
    def _is_at_me(event: GroupMessageEvent) -> bool:
        """检查群聊消息是否 @了机器人。"""
        bot_qq = str(event.self_id)
        for seg in event.message:
            if seg.type == "at" and seg.data.get("qq") == bot_qq:
                return True
        return False
