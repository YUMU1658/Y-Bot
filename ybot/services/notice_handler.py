"""通知事件处理服务。"""

from __future__ import annotations

from ybot.models.event import NoticeEvent, PokeNoticeEvent
from ybot.services.bot_info import BotInfoService
from ybot.services.message_sender import _next_poke_id
from ybot.storage.chat_log import ChatLogEntry, SessionChatLog


class NoticeHandler:
    """负责撤回和戳一戳通知写入 ChatLog。"""

    def __init__(self, bot_info: BotInfoService, chat_log: SessionChatLog) -> None:
        self._bot_info = bot_info
        self._chat_log = chat_log

    async def handle_recall(self, event: NoticeEvent) -> None:
        """处理撤回通知，标记 SessionChatLog 中对应消息为已撤回。"""
        if event.notice_type not in ("group_recall", "friend_recall"):
            return

        msg_id = event.raw_data.get("message_id")
        if msg_id is None:
            return

        hint = await self.build_recall_hint(event)
        self._chat_log.mark_recalled(int(msg_id), hint)

    async def build_recall_hint(self, event: NoticeEvent) -> str:
        """根据撤回事件构建人类可读的撤回提示。"""
        if event.notice_type == "friend_recall":
            return "对方撤回了这条消息"

        user_id = event.raw_data.get("user_id")
        operator_id = event.raw_data.get("operator_id")
        group_id = event.raw_data.get("group_id")

        if not operator_id or not group_id:
            return "已撤回"

        try:
            login_info = await self._bot_info.get_login_info()
            bot_id = login_info.user_id
        except Exception:
            bot_id = 0

        try:
            op_member = await self._bot_info.get_member_info(group_id, operator_id)
            op_name = op_member.card or op_member.nickname or str(operator_id)
        except Exception:
            op_name = str(operator_id)

        if operator_id == user_id:
            return f"{op_name}撤回了这条消息"
        if user_id == bot_id:
            return f"管理员{op_name}撤回了你的这条消息"
        return f"管理员{op_name}撤回了这条消息"

    async def handle_poke(self, event: PokeNoticeEvent) -> None:
        """处理戳一戳事件，统一写入 SessionChatLog。"""
        group_id = event.group_id
        login_info = await self._bot_info.get_login_info()
        is_bot = event.user_id == login_info.user_id
        target_is_bot = event.target_id == login_info.user_id

        if group_id:
            poker_member = await self._bot_info.get_member_info(group_id, event.user_id)
            poker_name = poker_member.card or poker_member.nickname or str(event.user_id)
        else:
            poker_member = None
            poker_name = event.raw_data.get("sender_nick", "") or str(event.user_id)

        if target_is_bot:
            target_display = f"你({login_info.user_id})"
        elif group_id:
            target_member = await self._bot_info.get_member_info(group_id, event.target_id)
            target_name = target_member.card or target_member.nickname or str(event.target_id)
            target_display = f"{target_name}({event.target_id})"
        else:
            target_display = str(event.target_id)

        poke_text = event.poke_text or "戳了戳"
        text = f"{poker_name}({event.user_id}) {poke_text} {target_display}"

        if group_id:
            session_key = f"group_{group_id}"
            nickname = poker_member.nickname or str(event.user_id) if poker_member else str(event.user_id)
            card = poker_member.card or "" if poker_member else ""
        else:
            session_key = f"friend_{event.user_id}"
            nickname = poker_name
            card = ""

        entry = ChatLogEntry(
            message_id=_next_poke_id(),
            session_key=session_key,
            user_id=event.user_id,
            nickname=nickname,
            card=card,
            role="",
            level="",
            title="",
            is_friend=False,
            timestamp=event.time,
            text=text,
            is_bot=is_bot,
            entry_type="poke",
        )
        self._chat_log.add(entry)
