"""消息上下文收集服务。"""

from __future__ import annotations

from ybot.core.config import AIConfig
from ybot.models.event import GroupMessageEvent, MessageEvent, PrivateMessageEvent
from ybot.services.env_builder import MessageFormatter
from ybot.services.message_content import MessageContentResolver
from ybot.storage.chat_log import ChatLogEntry, SessionChatLog
from ybot.storage.conversation import ConversationStore


class MessageContextCollector:
    """负责 ChatLog 收集、私聊 session key 推断和上下文消息构建。"""

    def __init__(
        self,
        ai_config: AIConfig,
        chat_log: SessionChatLog,
        conv_store: ConversationStore,
        content_resolver: MessageContentResolver,
    ) -> None:
        self._ai_config = ai_config
        self._chat_log = chat_log
        self._conv_store = conv_store
        self._content_resolver = content_resolver

    async def collect_message(
        self, event: MessageEvent, *, session_key: str, is_bot: bool
    ) -> None:
        """收集消息到会话上下文缓冲区。"""
        text = self._content_resolver.extract_content(event)
        if not text.strip():
            return

        text = await self._content_resolver.resolve_reply(text, event)
        text = await self._content_resolver.resolve_forward(text, event)

        if isinstance(event, GroupMessageEvent):
            nickname = event.sender.nickname or str(event.user_id)
            card = event.sender.card or ""
            role = event.sender.role or ""
        else:
            nickname = event.sender.nickname or str(event.user_id)
            card = ""
            role = ""

        entry = ChatLogEntry(
            message_id=event.message_id,
            session_key=session_key,
            user_id=event.user_id,
            nickname=nickname,
            card=card,
            role=role,
            level="",
            title="",
            is_friend=False,
            timestamp=event.time,
            text=text,
            is_bot=is_bot,
        )
        self._chat_log.add(entry)

    def resolve_bot_private_session_key(self, event: PrivateMessageEvent) -> str | None:
        """从 BOT 自身的私聊 message_sent 事件中推断 session_key。"""
        target_id = event.raw_data.get("target_id")
        if not target_id:
            target_id = event.raw_data.get("peer_id")
        if not target_id:
            return None

        if event.sub_type == "group":
            source_group_id = event.raw_data.get("sender", {}).get("group_id", 0)
            return f"temp_{source_group_id}_{target_id}"
        return f"friend_{target_id}"

    async def build_context_user_message(
        self,
        session_key: str,
        current_msg_id: int,
        formatted_msg: str,
    ) -> tuple[str, int | None]:
        """构建带参考聊天记录的 user 消息。"""
        prev_last_ref_id = await self._conv_store.get_last_ref_msg_id(session_key)
        ref_entries = self._chat_log.get_between(
            session_key=session_key,
            after_msg_id=prev_last_ref_id,
            before_msg_id=current_msg_id,
            limit=self._ai_config.context_limit,
        )

        if not session_key.startswith("group_"):
            ref_entries = [e for e in ref_entries if e.is_bot]

        last_ref_id = current_msg_id
        context_msg = MessageFormatter.build_context_message(
            ref_entries, formatted_msg, session_key=session_key
        )
        return context_msg, last_ref_id
