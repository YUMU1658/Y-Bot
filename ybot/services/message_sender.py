"""消息发送服务。

从 Bot 类中提取的消息发送职责，包括群聊/私聊消息发送、
戳一戳发送和 ChatLog 写入。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ybot.services.message_builder import text_to_segments
from ybot.storage.chat_log import ChatLogEntry
from ybot.utils.logger import get_logger

if TYPE_CHECKING:
    from ybot.core.ws_server import WebSocketServer
    from ybot.services.bot_info import BotInfoService
    from ybot.services.poke_limiter import PokeLimiter
    from ybot.storage.chat_log import SessionChatLog

logger = get_logger("消息发送")

# 戳一戳伪 message_id 计数器（负数，避免与正整数的真实 message_id 冲突）
_poke_id_counter: int = 0


def _next_poke_id() -> int:
    """生成下一个戳一戳伪 message_id（递减负数）。"""
    global _poke_id_counter
    _poke_id_counter -= 1
    return _poke_id_counter


class MessageSender:
    """消息发送服务。

    封装群聊/私聊消息发送、戳一戳发送和 ChatLog 写入逻辑，
    从 Bot 类中剥离以降低其职责复杂度。
    """

    def __init__(
        self,
        ws_server: WebSocketServer,
        bot_info: BotInfoService,
        chat_log: SessionChatLog,
        poke_limiter: PokeLimiter,
    ) -> None:
        self._ws_server = ws_server
        self._bot_info = bot_info
        self._chat_log = chat_log
        self._poke_limiter = poke_limiter

    async def send_group_msg(
        self,
        group_id: int,
        message: str,
        reply_id: int | None = None,
        *,
        session_key: str | None = None,
    ) -> int | None:
        """发送群聊消息并主动收集到 chat_log。

        将文本中的 <at qq="..."/> 标签解析为 OneBot at 消息段，
        以消息段数组格式发送。当 reply_id 不为 None 时，在消息段
        数组最前面插入 reply 类型段以实现引用回复。

        Args:
            group_id: 目标群号。
            message: 消息文本（可能包含 <at> 标签）。
            reply_id: 要引用的消息 ID（可选）。
            session_key: 会话标识（用于 chat_log 写入）。
                如果不提供，默认为 group_{group_id}。

        Returns:
            发送成功时返回 message_id，失败时返回 None。
        """
        segments = text_to_segments(message)
        if reply_id is not None:
            segments.insert(0, {"type": "reply", "data": {"id": str(reply_id)}})

        try:
            result = await self._ws_server.call_api(
                "send_group_msg",
                {"group_id": group_id, "message": segments},
            )
            msg_id = result.get("message_id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error(f"发送群聊消息失败: {e}")
            return None

        # 主动写入 chat_log
        if msg_id is not None:
            sk = session_key or f"group_{group_id}"
            login_info = await self._bot_info.get_login_info()
            entry = ChatLogEntry(
                message_id=msg_id,
                session_key=sk,
                user_id=login_info.user_id,
                nickname=login_info.nickname or str(login_info.user_id),
                card="",
                role="",
                level="",
                title="",
                is_friend=False,
                timestamp=time.time(),
                text=message,
                is_bot=True,
            )
            self._chat_log.add(entry)

        return msg_id

    async def send_private_msg(
        self,
        user_id: int,
        message: str,
        reply_id: int | None = None,
        *,
        session_key: str | None = None,
    ) -> int | None:
        """发送私聊消息并主动收集到 chat_log。

        将文本中的 <at qq="..."/> 标签解析为 OneBot at 消息段，
        以消息段数组格式发送。当 reply_id 不为 None 时，在消息段
        数组最前面插入 reply 类型段以实现引用回复。

        Args:
            user_id: 目标用户 QQ 号。
            message: 消息文本（可能包含 <at> 标签）。
            reply_id: 要引用的消息 ID（可选）。
            session_key: 会话标识（用于 chat_log 写入）。
                如果不提供，默认为 friend_{user_id}。

        Returns:
            发送成功时返回 message_id，失败时返回 None。
        """
        segments = text_to_segments(message)
        if reply_id is not None:
            segments.insert(0, {"type": "reply", "data": {"id": str(reply_id)}})

        try:
            result = await self._ws_server.call_api(
                "send_private_msg",
                {"user_id": user_id, "message": segments},
            )
            msg_id = result.get("message_id") if isinstance(result, dict) else None
        except Exception as e:
            logger.error(f"发送私聊消息失败: {e}")
            return None

        # 主动写入 chat_log
        if msg_id is not None:
            sk = session_key or f"friend_{user_id}"
            login_info = await self._bot_info.get_login_info()
            entry = ChatLogEntry(
                message_id=msg_id,
                session_key=sk,
                user_id=login_info.user_id,
                nickname=login_info.nickname or str(login_info.user_id),
                card="",
                role="",
                level="",
                title="",
                is_friend=False,
                timestamp=time.time(),
                text=message,
                is_bot=True,
            )
            self._chat_log.add(entry)

        return msg_id

    async def send_poke(
        self, target_id: int, group_id: int | None = None
    ) -> tuple[bool, str]:
        """发送戳一戳动作。

        Args:
            target_id: 被戳用户的 QQ 号。
            group_id: 群号（群聊时提供，私聊时为 None）。

        Returns:
            (成功与否, 互动文案或失败原因)
        """
        # 1. 速率限制检查
        reject_reason = self._poke_limiter.check(target_id)
        if reject_reason:
            return False, reject_reason

        # 2. 调用 OneBot API（优先使用 group_poke/friend_poke，失败后降级到 send_poke）
        try:
            if group_id:
                try:
                    await self._ws_server.call_api(
                        "group_poke",
                        {"group_id": group_id, "user_id": target_id},
                        timeout=5.0,
                    )
                except Exception:
                    # 降级到 send_poke
                    await self._ws_server.call_api(
                        "send_poke",
                        {"group_id": group_id, "user_id": target_id},
                        timeout=5.0,
                    )
            else:
                try:
                    await self._ws_server.call_api(
                        "friend_poke",
                        {"user_id": target_id},
                        timeout=5.0,
                    )
                except Exception:
                    # 降级到 send_poke
                    await self._ws_server.call_api(
                        "send_poke",
                        {"user_id": target_id},
                        timeout=5.0,
                    )
        except Exception as e:
            return False, f"戳一戳发送失败: {e}"

        # 3. 记录成功
        self._poke_limiter.record(target_id)

        # 4. 构建互动文案
        login_info = await self._bot_info.get_login_info()
        bot_name = login_info.nickname or str(login_info.user_id)
        poke_text = f"{bot_name}({login_info.user_id}) 戳了戳 {target_id}"

        return True, poke_text

    async def write_poke_chat_log(
        self, *, session_key: str, poke_text: str
    ) -> None:
        """将 bot 主动发起的戳一戳写入 ChatLog。

        Args:
            session_key: 会话标识。
            poke_text: 互动文案（如 "Bot名(QQ号) 戳了戳 目标QQ号"）。
        """
        login_info = await self._bot_info.get_login_info()
        entry = ChatLogEntry(
            message_id=_next_poke_id(),
            session_key=session_key,
            user_id=login_info.user_id,
            nickname=login_info.nickname or str(login_info.user_id),
            card="",
            role="",
            level="",
            title="",
            is_friend=False,
            timestamp=time.time(),
            text=poke_text,
            is_bot=True,
            entry_type="poke",
        )
        self._chat_log.add(entry)
