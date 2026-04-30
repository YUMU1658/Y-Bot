"""撤回消息工具实现。

支持撤回 bot 自己的消息和管理员撤回他人消息，
包含完整的前置校验和错误处理。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ybot.tools.base import BaseTool, ToolContext, ToolResult
from ybot.utils.logger import get_logger

logger = get_logger("撤回工具")

# bot 自身消息撤回时限（秒）
_SELF_RECALL_LIMIT = 120


class RecallMsgTool(BaseTool):
    """撤回消息工具。

    通过 OneBot API 撤回指定的消息。支持：
    - 撤回 bot 自己发送的消息（2 分钟内）
    - 管理员/群主撤回群成员的消息
    """

    @property
    def name(self) -> str:
        return "recall_msg"

    @property
    def description(self) -> str:
        return (
            "撤回指定的消息。可以撤回自己发送的消息（2分钟内），"
            "或以管理员身份撤回群成员的消息。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "要撤回的消息ID列表",
                },
            },
            "required": ["message_ids"],
        }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """执行撤回操作。

        对每个 message_id 依次执行前置校验和撤回，
        多个消息之间间隔 0.5 秒。

        Args:
            arguments: 包含 ``message_ids`` 列表的参数字典。
            context: 工具执行上下文。

        Returns:
            撤回结果。
        """
        message_ids: list[int] = arguments.get("message_ids", [])
        if not message_ids:
            return ToolResult(success=False, message="未提供要撤回的消息ID")

        # 获取 bot 自身 ID
        login_info = await context.bot_info.get_login_info()
        bot_id = login_info.user_id

        results: list[tuple[int, bool, str]] = []  # (msg_id, success, detail)

        for i, msg_id in enumerate(message_ids):
            success, detail = await self._recall_single(
                msg_id, bot_id, context
            )
            results.append((msg_id, success, detail))

            # 多个消息之间间隔 0.5 秒（最后一个不等待）
            if i < len(message_ids) - 1:
                await asyncio.sleep(0.5)

        return self._format_results(results)

    async def _recall_single(
        self,
        msg_id: int,
        bot_id: int,
        context: ToolContext,
    ) -> tuple[bool, str]:
        """撤回单条消息。

        Args:
            msg_id: 消息 ID。
            bot_id: Bot 自身的 QQ 号。
            context: 工具执行上下文。

        Returns:
            (是否成功, 详情描述)
        """
        # 1. 获取消息详情
        try:
            msg_data = await context.ws_server.call_api(
                "get_msg", {"message_id": msg_id}, timeout=5.0
            )
        except Exception as e:
            logger.debug(f"get_msg 失败 (msg_id={msg_id}): {e}")
            return False, f"消息ID {msg_id} 不存在或已过期"

        if not msg_data:
            return False, f"消息ID {msg_id} 不存在或已过期"

        # 2. 提取消息信息
        sender_id = msg_data.get("sender", {}).get("user_id") or msg_data.get(
            "user_id", 0
        )
        msg_time = msg_data.get("time", 0)
        message_type = msg_data.get("message_type", "")
        group_id = msg_data.get("group_id")

        # 提取发送者昵称（用于撤回提示）
        sender_name = msg_data.get("sender", {}).get("nickname", str(sender_id))

        is_own_msg = sender_id == bot_id

        # 3. 前置校验
        validation_error = await self._validate(
            msg_id=msg_id,
            is_own_msg=is_own_msg,
            msg_time=msg_time,
            message_type=message_type,
            group_id=group_id,
            sender_id=sender_id,
            bot_id=bot_id,
            context=context,
        )
        if validation_error:
            return False, validation_error

        # 4. 调用 delete_msg API
        try:
            await context.ws_server.call_api(
                "delete_msg", {"message_id": msg_id}, timeout=5.0
            )
        except Exception as e:
            error_str = str(e)
            logger.warning(f"delete_msg 失败 (msg_id={msg_id}): {error_str}")
            # 尝试从错误信息中提取有用信息
            if "1400" in error_str or "1401" in error_str:
                return False, f"无法撤回消息 #{msg_id}：权限不足或消息已过期"
            if "1404" in error_str:
                return False, f"消息 #{msg_id} 不存在"
            return False, f"撤回消息 #{msg_id} 失败: {error_str}"

        # 5. 成功后标记 chat_log
        if is_own_msg:
            hint = "你撤回了这条消息"
        else:
            hint = f"你撤回了{sender_name}的这条消息"

        context.chat_log.mark_recalled(msg_id, hint)
        logger.info(f"已撤回消息 #{msg_id} (sender={sender_id})")

        return True, "已撤回"

    async def _validate(
        self,
        *,
        msg_id: int,
        is_own_msg: bool,
        msg_time: int,
        message_type: str,
        group_id: int | None,
        sender_id: int,
        bot_id: int,
        context: ToolContext,
    ) -> str | None:
        """前置校验。

        Returns:
            校验失败时返回错误描述，通过时返回 None。
        """
        # bot 自己的消息：检查 2 分钟时限
        if is_own_msg:
            elapsed = time.time() - msg_time
            if elapsed > _SELF_RECALL_LIMIT:
                return f"无法撤回消息 #{msg_id}：已超过2分钟时限"
            return None  # 自己的消息在时限内，无需其他检查

        # 他人的消息
        # 私聊中只能撤回自己的消息
        if message_type == "private":
            return f"无法撤回消息 #{msg_id}：私聊中只能撤回自己的消息"

        # 群聊中撤回他人消息需要管理员权限
        if not group_id:
            return f"无法撤回消息 #{msg_id}：无法确定消息所在群"

        # 获取 bot 在群中的角色
        bot_member = await context.bot_info.get_member_info(group_id, bot_id)
        bot_role = bot_member.role  # "owner" / "admin" / "member"

        if bot_role == "member":
            return (
                f"无法撤回消息 #{msg_id}：权限不足，"
                "你当前是普通成员，无法撤回他人消息"
            )

        # bot 是管理员（非群主）时，不能撤回群主或其他管理员的消息
        if bot_role == "admin":
            sender_member = await context.bot_info.get_member_info(
                group_id, sender_id
            )
            sender_role = sender_member.role
            if sender_role in ("owner", "admin"):
                return (
                    f"无法撤回消息 #{msg_id}："
                    "无法撤回同级或更高权限者的消息"
                )

        return None  # 校验通过

    @staticmethod
    def _format_results(
        results: list[tuple[int, bool, str]],
    ) -> ToolResult:
        """格式化撤回结果。

        Args:
            results: (msg_id, success, detail) 列表。

        Returns:
            格式化后的 ToolResult。
        """
        total = len(results)
        success_count = sum(1 for _, ok, _ in results if ok)
        success_ids = [str(mid) for mid, ok, _ in results if ok]

        if success_count == total:
            # 全部成功
            ids_str = ", ".join(f"#{mid}" for mid, _, _ in results)
            return ToolResult(
                success=True,
                message=f"已成功撤回 {total} 条消息（{ids_str}）",
            )

        if success_count == 0:
            # 全部失败
            lines = [f"- #{mid}：{detail}" for mid, _, detail in results]
            return ToolResult(
                success=False,
                message="撤回失败：\n" + "\n".join(lines),
            )

        # 部分成功
        lines: list[str] = []
        for mid, ok, detail in results:
            if ok:
                lines.append(f"- #{mid}：已撤回")
            else:
                lines.append(f"- #{mid}：失败，{detail}")
        return ToolResult(
            success=False,
            message="撤回结果：\n" + "\n".join(lines),
        )
