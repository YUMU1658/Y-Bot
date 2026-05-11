"""/clear 指令实现。

清除对话数据上下文，支持清除当前会话、所有会话或指定会话。
"""

from __future__ import annotations

import re

from ybot.commands.base import BaseCommand, CommandContext, CommandResult
from ybot.storage.chat_log import SessionChatLog
from ybot.storage.conversation import ConversationStore


class ClearCommand(BaseCommand):
    """清除对话数据上下文指令。

    支持的格式：
    - ``/clear`` — 清除当前会话
    - ``/clear all`` — 清除所有会话
    - ``/clear group_123456789`` — 清除指定会话
    - ``/clear group_123,friend_456`` — 逗号分隔清除多个会话
    """

    def __init__(
        self, conv_store: ConversationStore, chat_log: SessionChatLog
    ) -> None:
        self._conv_store = conv_store
        self._chat_log = chat_log

    @property
    def name(self) -> str:
        return "clear"

    @property
    def description(self) -> str:
        return "清除对话数据上下文"

    async def execute(self, ctx: CommandContext) -> CommandResult:
        """执行清除指令。"""
        args = ctx.raw_args.strip()

        if not args:
            # /clear — 清除当前会话
            await self._conv_store.clear_session(ctx.session_key)
            self._chat_log.clear(ctx.session_key)
            return CommandResult(
                success=True,
                message=f"已清除当前会话 ({ctx.session_key}) 的对话上下文。",
            )

        if args.lower() == "all":
            # /clear all — 清除所有会话
            await self._conv_store.clear_all_sessions()
            self._chat_log.clear_all()
            return CommandResult(
                success=True,
                message="已清除所有会话的对话上下文。",
            )

        # /clear session_key1,session_key2,...
        keys = [k.strip() for k in args.split(",") if k.strip()]
        valid_keys: list[str] = []
        invalid_keys: list[str] = []
        for key in keys:
            if self._is_valid_session_key(key):
                valid_keys.append(key)
            else:
                invalid_keys.append(key)

        for key in valid_keys:
            await self._conv_store.clear_session(key)
            self._chat_log.clear(key)

        # 构建结果消息
        parts: list[str] = []
        if valid_keys:
            parts.append(f"已清除 {len(valid_keys)} 个会话: {', '.join(valid_keys)}")
        if invalid_keys:
            parts.append(f"无效的会话标识: {', '.join(invalid_keys)}")

        return CommandResult(
            success=len(invalid_keys) == 0,
            message="\n".join(parts),
        )

    @staticmethod
    def _is_valid_session_key(key: str) -> bool:
        """验证 session_key 格式。

        合法格式：
        - ``group_<数字>``
        - ``friend_<数字>``
        - ``temp_<数字>_<数字>``
        """
        return bool(re.match(r"^(group_\d+|friend_\d+|temp_\d+_\d+)$", key))
