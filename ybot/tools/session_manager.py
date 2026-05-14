"""会话管理工具。

提供会话列表查看功能，支持群聊、私聊和临时会话。
主数据源为 NapCat ``get_recent_contact`` API，不可用时自动降级到内部 chat_log。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from ybot.constants import TZ_CST
from ybot.models.message import MessageSegment, parse_message, segment_to_text
from ybot.tools.base import BaseTool, ToolContext, ToolResult
from ybot.utils.logger import get_logger

logger = get_logger("会话管理工具")

# ── 常量 ──

CHAT_TYPE_LABELS: dict[int, str] = {
    1: "私聊",
    2: "群聊",
}

DEFAULT_COUNT = 20
MAX_COUNT = 50
HISTORY_SCAN_COUNT = 20  # 用于 @bot 检测的历史消息扫描数量
PREVIEW_COUNT = 2  # 每个会话显示的预览消息数


class SessionManagerTool(BaseTool):
    """会话管理工具 — 查看最近会话列表。"""

    @property
    def name(self) -> str:
        return "session_manager"

    @property
    def description(self) -> str:
        return (
            "会话管理工具。查看最近的聊天会话列表，包括群聊、私聊和临时会话。\n"
            "每个会话显示最新2条消息预览，并标注是否有人@bot或引用bot消息。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["recent_sessions"],
                    "description": "操作类型。recent_sessions：获取最近会话列表。",
                },
                "count": {
                    "type": "integer",
                    "description": "获取的会话数量，默认 20，最大 50。",
                },
                "offset": {
                    "type": "integer",
                    "description": "偏移量，用于翻页。默认 0 表示从最新开始。",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """执行会话管理操作。

        Args:
            arguments: 包含 ``action`` 及相关参数的字典。
            context: 工具执行上下文。

        Returns:
            操作结果。
        """
        action = arguments.get("action", "")
        match action:
            case "recent_sessions":
                return await self._handle_recent_sessions(arguments, context)
            case _:
                return ToolResult(success=False, message=f"未知操作: {action}")

    # ── recent_sessions 主流程 ──

    async def _handle_recent_sessions(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """获取最近会话列表。

        优先使用 NapCat ``get_recent_contact`` API，不可用时降级到内部 chat_log。

        Args:
            arguments: 工具调用参数（可能包含 count/offset）。
            context: 工具执行上下文。

        Returns:
            格式化的会话列表结果。
        """
        count = max(1, min(int(arguments.get("count", DEFAULT_COUNT)), MAX_COUNT))
        offset = max(int(arguments.get("offset", 0)), 0)

        login_info = await context.bot_info.get_login_info()
        bot_id = login_info.user_id

        # 尝试 NapCat API
        try:
            contacts = await self._fetch_recent_contacts(
                offset + count, context
            )
            return await self._process_napcat_contacts(
                contacts, offset, count, bot_id, context
            )
        except Exception as e:
            logger.info(f"get_recent_contact 不可用，使用降级模式: {e}")

        # 降级：从内部数据构建
        return await self._process_fallback(offset, count, bot_id, context)

    # ── NapCat 模式 ──

    async def _fetch_recent_contacts(
        self, count: int, context: ToolContext
    ) -> list[dict[str, Any]]:
        """调用 ``get_recent_contact`` API。

        Args:
            count: 请求的会话数量。
            context: 工具执行上下文。

        Returns:
            会话列表（每项为 NapCat 返回的会话 dict）。

        Raises:
            RuntimeError: API 返回格式异常。
        """
        result = await context.ws_server.call_api(
            "get_recent_contact", {"count": count}
        )
        if not isinstance(result, list):
            raise RuntimeError("get_recent_contact 返回格式异常")
        return result

    async def _process_napcat_contacts(
        self,
        contacts: list[dict[str, Any]],
        offset: int,
        count: int,
        bot_id: int,
        context: ToolContext,
    ) -> ToolResult:
        """处理 NapCat API 返回的会话列表。

        Args:
            contacts: ``get_recent_contact`` 返回的原始会话列表。
            offset: 分页偏移量。
            count: 分页数量。
            bot_id: Bot 的 QQ 号。
            context: 工具执行上下文。

        Returns:
            格式化的会话列表结果。
        """
        # 1. 按 msgTime 降序排序（新→旧）
        contacts.sort(
            key=lambda c: int(c.get("msgTime", "0")), reverse=True
        )
        total = len(contacts)

        # 2. 分页
        page = contacts[offset : offset + count]
        if not page:
            return ToolResult(success=True, message="没有更多会话。")

        # 3. 并发获取每个会话的历史消息
        tasks = [self._fetch_session_messages(s, context) for s in page]
        msg_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. 构建输出
        lines: list[str] = [
            f"最近会话列表（共 {total} 个），"
            f"当前显示第 {offset + 1}~{offset + len(page)} 个：\n"
        ]

        for i, (session, msgs) in enumerate(zip(page, msg_results)):
            if isinstance(msgs, Exception):
                msgs = []
                latest = session.get("lastestMsg")
                if latest:
                    msgs = [latest]

            entry = await self._build_session_entry(
                i + offset + 1, session, msgs, bot_id, context
            )
            lines.append(entry)

        # 5. 分页提示
        remaining = total - offset - len(page)
        if remaining > 0:
            lines.append(
                f"\n提示：还有 {remaining} 个会话未显示。"
                "可使用 offset 和 count 参数查看更多。"
            )

        return ToolResult(success=True, message="\n".join(lines))

    async def _fetch_session_messages(
        self, session: dict[str, Any], context: ToolContext
    ) -> list[dict[str, Any]]:
        """获取会话的最近消息（用于预览和 @bot 检测）。

        Args:
            session: 单个会话的 NapCat 数据。
            context: 工具执行上下文。

        Returns:
            消息列表（OneBot v11 消息格式）。
        """
        chat_type = session.get("chatType", 0)
        peer_id = session.get("peerUin", "")

        try:
            if chat_type == 2:  # 群聊
                result = await context.ws_server.call_api(
                    "get_group_msg_history",
                    {"group_id": int(peer_id)},
                )
                # get_group_msg_history 返回 {"messages": [...]}
                if isinstance(result, dict):
                    return result.get("messages", [])
                return []
            else:  # 私聊
                result = await context.ws_server.call_api(
                    "get_friend_msg_history",
                    {"user_id": str(peer_id), "count": HISTORY_SCAN_COUNT},
                )
                if isinstance(result, dict):
                    return result.get("messages", [])
                return []
        except Exception as e:
            logger.warning(f"获取会话 {peer_id} 历史消息失败: {e}")
            # 降级：使用 get_recent_contact 中的 lastestMsg
            latest = session.get("lastestMsg")
            return [latest] if latest else []

    async def _build_session_entry(
        self,
        index: int,
        session: dict[str, Any],
        messages: list[dict[str, Any]],
        bot_id: int,
        context: ToolContext,
    ) -> str:
        """构建单个会话条目的格式化文本。

        Args:
            index: 序号（1-based）。
            session: 会话的 NapCat 数据。
            messages: 该会话的历史消息列表。
            bot_id: Bot 的 QQ 号。
            context: 工具执行上下文。

        Returns:
            格式化的会话条目文本。
        """
        # 1. 会话类型和名称
        chat_type = session.get("chatType", 0)
        peer_name = session.get("peerName", "未知")
        peer_id = session.get("peerUin", "?")
        type_label = CHAT_TYPE_LABELS.get(chat_type, "未知")

        # 2. @bot 检测（扫描全部历史消息）
        mention_tags = await self._check_bot_mentions(messages, bot_id, context)
        tag_str = ""
        if "at-me" in mention_tags:
            tag_str += "【⚡🔔@ME】"
        if "reply-me" in mention_tags:
            tag_str += "【⚡💬RE】"

        # 3. 标题行
        header = f"{index}. {tag_str}{peer_name}({peer_id}) [{type_label}]"

        # 4. 消息预览（最新 2 条）
        is_group = chat_type == 2
        preview_msgs = messages[-PREVIEW_COUNT:] if messages else []
        preview_lines: list[str] = []
        for j, msg in enumerate(reversed(preview_msgs)):
            # reversed: 最新的在前
            prefix = "├" if j < len(preview_msgs) - 1 else "└"
            text, time_str = self._format_message_preview(msg, is_group, bot_id)
            preview_lines.append(f"   {prefix} {text}    {time_str}")

        if preview_lines:
            return header + "\n" + "\n".join(preview_lines)
        return header

    # ── 降级模式 ──

    async def _process_fallback(
        self,
        offset: int,
        count: int,
        bot_id: int,
        context: ToolContext,
    ) -> ToolResult:
        """从内部 chat_log 构建会话列表（降级模式）。

        当 NapCat ``get_recent_contact`` API 不可用时使用。
        仅包含 bot 运行期间收到的消息。

        Args:
            offset: 分页偏移量。
            count: 分页数量。
            bot_id: Bot 的 QQ 号。
            context: 工具执行上下文。

        Returns:
            格式化的会话列表结果。
        """
        # 1. 获取所有活跃 session
        all_keys = context.chat_log.get_all_session_keys()
        if not all_keys:
            return ToolResult(
                success=True,
                message="当前没有活跃会话记录（降级模式：仅显示 Bot 运行期间的活跃会话）。",
            )

        # 2. 对每个 session 获取最近消息，按最新时间排序
        session_data: list[tuple[str, list[Any]]] = []
        for sk in all_keys:
            entries = context.chat_log.get_recent(sk, limit=HISTORY_SCAN_COUNT)
            if entries:
                session_data.append((sk, entries))

        session_data.sort(key=lambda x: x[1][-1].timestamp, reverse=True)
        total = len(session_data)

        # 3. 分页
        page = session_data[offset : offset + count]
        if not page:
            return ToolResult(success=True, message="没有更多会话。")

        # 4. 格式化
        lines: list[str] = [
            f"最近会话列表（共 {total} 个），"
            f"当前显示第 {offset + 1}~{offset + len(page)} 个：",
            "（降级模式：仅显示 Bot 运行期间的活跃会话）\n",
        ]

        for i, (sk, entries) in enumerate(page):
            # 从 session_key 推断类型
            type_label, display_name = await self._parse_session_key(
                sk, context
            )

            # @bot 检测（从 ChatLogEntry）
            mention_tags = self._check_bot_mentions_from_entries(entries, bot_id)
            tag_str = ""
            if "at-me" in mention_tags:
                tag_str += "【⚡🔔@ME】"
            if "reply-me" in mention_tags:
                tag_str += "【⚡💬RE】"

            header = f"{i + offset + 1}. {tag_str}{display_name} [{type_label}]"

            # 预览最新 2 条
            preview = entries[-PREVIEW_COUNT:]
            preview_lines: list[str] = []
            for j, entry in enumerate(reversed(preview)):
                prefix = "├" if j < len(preview) - 1 else "└"
                name = entry.card or entry.nickname
                if entry.is_bot:
                    name = "我"
                time_str = _format_time_short(entry.timestamp)
                # entry.text 已包含占位标签，截断过长文本
                text = (
                    entry.text[:50] + "..."
                    if len(entry.text) > 50
                    else entry.text
                )
                preview_lines.append(f"   {prefix} {name}: {text}    {time_str}")

            if preview_lines:
                lines.append(header + "\n" + "\n".join(preview_lines))
            else:
                lines.append(header)

        remaining = total - offset - len(page)
        if remaining > 0:
            lines.append(
                f"\n提示：还有 {remaining} 个会话未显示。"
                "可使用 offset 和 count 参数查看更多。"
            )

        return ToolResult(success=True, message="\n".join(lines))

    async def _parse_session_key(
        self, session_key: str, context: ToolContext
    ) -> tuple[str, str]:
        """从 session_key 推断会话类型和显示名称。

        Args:
            session_key: 会话标识（如 ``group_12345``）。
            context: 工具执行上下文。

        Returns:
            (类型标签, 显示名称) 元组。
        """
        if session_key.startswith("group_"):
            group_id = int(session_key[6:])
            try:
                group_info = await context.bot_info.get_group_info(group_id)
                name = group_info.group_name or f"群{group_id}"
            except Exception:
                name = f"群{group_id}"
            return "群聊", f"{name}({group_id})"
        elif session_key.startswith("friend_"):
            user_id = int(session_key[7:])
            return "私聊", f"好友({user_id})"
        elif session_key.startswith("temp_"):
            return "临时会话", f"临时会话({session_key[5:]})"
        return "未知", session_key

    # ── 共用辅助方法 ──

    def _format_message_preview(
        self, msg: dict[str, Any], is_group: bool, bot_id: int
    ) -> tuple[str, str]:
        """格式化单条消息预览。

        Args:
            msg: OneBot v11 消息格式的消息 dict。
            is_group: 是否为群聊消息。
            bot_id: Bot 的 QQ 号。

        Returns:
            (内容行, 时间字符串) 元组。
        """
        sender = msg.get("sender", {})
        nickname = sender.get("nickname", "未知")
        card = sender.get("card", "")  # 群名片
        user_id = sender.get("user_id", 0)

        # 群聊显示群名片，无群名片则显示昵称；bot 自己显示为"我"
        if user_id == bot_id:
            display_name = "我"
        elif is_group and card:
            display_name = card
        else:
            display_name = nickname

        # 消息内容转为预览文本
        segments = parse_message(msg.get("message", []))
        preview_text = _preview_segments_text(segments, max_chars=50)

        # 时间格式化
        time_val = msg.get("time", 0)
        time_str = _format_time_short(time_val)

        return f"{display_name}: {preview_text}", time_str

    async def _check_bot_mentions(
        self,
        messages: list[dict[str, Any]],
        bot_id: int,
        context: ToolContext,
    ) -> set[str]:
        """从 OneBot 消息列表中检测 @bot / 引用 bot。

        基于消息段结构化数据检测，不可被纯文本伪造。

        Args:
            messages: OneBot v11 格式的消息列表。
            bot_id: Bot 的 QQ 号。
            context: 工具执行上下文。

        Returns:
            标签集合，如 ``{"at-me", "reply-me"}``。
        """
        tags: set[str] = set()
        reply_msg_ids: list[str] = []

        for msg in messages:
            sender_id = msg.get("sender", {}).get("user_id", 0)
            # 跳过 bot 自己发的消息
            if sender_id == bot_id:
                continue

            segments = parse_message(msg.get("message", []))
            for seg in segments:
                # 检测 @bot（基于消息段结构，非文本匹配）
                if seg.type == "at" and str(seg.data.get("qq")) == str(bot_id):
                    tags.add("at-me")

                # 收集 reply 消息 ID
                if seg.type == "reply":
                    reply_id = seg.data.get("id")
                    if reply_id:
                        reply_msg_ids.append(str(reply_id))

        # 批量检测 reply 目标是否为 bot 消息
        if reply_msg_ids and "reply-me" not in tags:
            # 先从内存 chat_log 查找
            unchecked_ids: list[str] = []
            for rid in reply_msg_ids:
                try:
                    entry = context.chat_log.get_by_id(int(rid))
                    if entry and entry.is_bot:
                        tags.add("reply-me")
                        break
                except (ValueError, TypeError):
                    pass
                else:
                    if "reply-me" not in tags:
                        unchecked_ids.append(rid)

            # chat_log 中未命中的，通过 get_msg API 查找（限制并发数）
            if "reply-me" not in tags and unchecked_ids:
                # 最多检查 5 条，避免过多 API 调用
                check_ids = unchecked_ids[:5]
                api_tasks = [
                    context.ws_server.call_api(
                        "get_msg", {"message_id": int(rid)}
                    )
                    for rid in check_ids
                ]
                api_results = await asyncio.gather(
                    *api_tasks, return_exceptions=True
                )
                for result in api_results:
                    if isinstance(result, Exception):
                        continue
                    if (
                        isinstance(result, dict)
                        and result.get("sender", {}).get("user_id") == bot_id
                    ):
                        tags.add("reply-me")
                        break

        return tags

    @staticmethod
    def _check_bot_mentions_from_entries(
        entries: list[Any], bot_id: int
    ) -> set[str]:
        """从 ChatLogEntry 列表中检测 @bot / 引用 bot（降级模式）。

        通过检查 ``entry.text`` 中的结构化标记来检测。
        ``ChatLogEntry.text`` 由 ``segments_to_content()`` 生成，
        其中 at 段格式为 ``<at qq="bot_id"/>``。

        Args:
            entries: ChatLogEntry 列表。
            bot_id: Bot 的 QQ 号。

        Returns:
            标签集合。
        """
        tags: set[str] = set()
        at_marker = f'<at qq="{bot_id}"/>'

        for entry in entries:
            # 跳过 bot 自己的消息
            if entry.is_bot:
                continue

            # 检测 @bot（segments_to_content 中 at 段格式为 <at qq="xxx"/>）
            if at_marker in entry.text:
                tags.add("at-me")

            # 检测引用 bot 消息：
            # segments_to_content 中 reply 段格式为 [回复:#msg_id]
            # 降级模式下无法直接确认被引用消息是否为 bot 发送，
            # 但可以通过 chat_log 的 _msg_index 间接判断。
            # 此处简化处理：不在降级模式中检测 reply-me，
            # 因为需要额外的消息查找逻辑且降级模式本身数据有限。

        return tags


# ── 模块级辅助函数 ──


def _format_time_short(ts: int | float) -> str:
    """将 Unix 时间戳格式化为 ``HH:MM:SS`` 格式。

    Args:
        ts: Unix 时间戳（秒）。

    Returns:
        格式化的时间字符串，无效时返回 ``"??:??:??"``。
    """
    if not ts:
        return "??:??:??"
    try:
        dt = datetime.fromtimestamp(ts, tz=TZ_CST)
        return dt.strftime("%H:%M:%S")
    except (OSError, ValueError):
        return "??:??:??"


def _preview_segments_text(
    segments: list[MessageSegment], *, max_chars: int = 50
) -> str:
    """将消息段列表转为截断的预览文本。

    Args:
        segments: 消息段列表。
        max_chars: 预览文本最大字符数。

    Returns:
        截断后的预览文本。
    """
    raw = "".join(segment_to_text(seg) for seg in segments)
    raw = raw.strip()
    if len(raw) > max_chars:
        return raw[:max_chars] + "..."
    return raw or "[空消息]"
