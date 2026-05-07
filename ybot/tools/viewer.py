"""查看工具实现。

支持查看消息完整文字内容、消息中的图片/表情包、以及用户/群头像。
与引用消息的 80 字符截断互补——本工具可获取完整内容。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ybot.models.message import parse_message, segments_to_content
from ybot.tools.base import BaseTool, ToolContext, ToolResult
from ybot.utils.logger import get_logger

logger = get_logger("查看工具")

# 东八区时区
_TZ_CST = timezone(timedelta(hours=8))

# 用户头像 URL 模板
_USER_AVATAR_URL = "https://q.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
# 群头像 URL 模板
_GROUP_AVATAR_URL = "https://p.qlogo.cn/gh/{group_id}/{group_id}/640/"


def _format_timestamp(ts: int) -> str:
    """将 Unix 时间戳格式化为可读时间。"""
    if not ts:
        return "未知"
    try:
        dt = datetime.fromtimestamp(ts, tz=_TZ_CST)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError):
        return "未知"


class ViewerTool(BaseTool):
    """查看工具。

    通过 OneBot API 查看消息完整内容、消息中的图片、以及用户/群头像。
    支持三种操作：view_message、view_message_image、view_avatar。
    """

    @property
    def name(self) -> str:
        return "viewer"

    @property
    def description(self) -> str:
        return (
            "查看消息内容或头像图片。支持三种操作：\n"
            '1. "view_message" — 通过消息ID获取消息的完整文字内容'
            "（发送者、时间、完整消息文本），"
            "适用于查看被截断的引用消息的完整内容。需要参数：message_id\n"
            '2. "view_message_image" — 通过消息ID查看消息中的图片、'
            "自定义表情、商城表情等图片内容，"
            "适用于查看参考聊天记录中的图片/表情包。需要参数：message_id\n"
            '3. "view_avatar" — 查看QQ用户或群的头像图片，'
            "可查看好友/群友/陌生人的头像。"
            '需要参数：avatar_type（"user"或"group"）、target_id（QQ号或群号）'
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["view_message", "view_message_image", "view_avatar"],
                    "description": "要执行的操作",
                },
                "message_id": {
                    "type": "integer",
                    "description": (
                        "要查看的消息ID"
                        "（view_message 和 view_message_image 操作需要）"
                    ),
                },
                "avatar_type": {
                    "type": "string",
                    "enum": ["user", "group"],
                    "description": (
                        "头像类型：user=QQ用户头像，group=群头像"
                        "（仅 view_avatar 操作需要）"
                    ),
                },
                "target_id": {
                    "type": "integer",
                    "description": (
                        "QQ号或群号（仅 view_avatar 操作需要）"
                    ),
                },
            },
            "required": ["action"],
        }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """执行查看操作。

        Args:
            arguments: 包含 ``action`` 及相关参数的字典。
            context: 工具执行上下文。

        Returns:
            查看结果。
        """
        action = arguments.get("action", "")
        if action not in ("view_message", "view_message_image", "view_avatar"):
            return ToolResult(
                success=False,
                message=f"未知操作: {action}",
            )

        if action == "view_message":
            return await self._handle_view_message(arguments, context)
        if action == "view_message_image":
            return await self._handle_view_message_image(arguments, context)
        # action == "view_avatar"
        return await self._handle_view_avatar(arguments, context)

    # ──────────────────────────────────────────────
    #  view_message
    # ──────────────────────────────────────────────

    async def _handle_view_message(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """处理 ``view_message`` 操作。

        通过 get_msg API 获取消息详情，解析消息段并返回完整文字内容。
        不涉及图片，不依赖 enable_vision。

        Args:
            arguments: 工具调用参数（需要 message_id）。
            context: 工具执行上下文。

        Returns:
            包含完整消息内容的结果。
        """
        message_id = arguments.get("message_id")
        if message_id is None:
            return ToolResult(
                success=False,
                message="查看消息内容需要提供 message_id 参数",
            )

        # 获取消息详情
        data = await self._fetch_message(int(message_id), context)
        if data is None:
            return ToolResult(
                success=False,
                message=f"获取消息失败: 消息不存在或已过期 (message_id={message_id})",
            )

        # 提取发送者信息
        sender = data.get("sender", {})
        nickname = sender.get("nickname", "?")
        user_id = data.get("user_id", "?")

        # 时间戳格式化
        timestamp = data.get("time", 0)
        time_str = _format_timestamp(timestamp)

        # 解析消息段，生成完整文本内容（不截断）
        raw_message_data = data.get("message", [])
        if isinstance(raw_message_data, list) and raw_message_data:
            segments = parse_message(raw_message_data)
            full_content = segments_to_content(segments)
        else:
            # 降级：使用 raw_message 字段
            full_content = data.get("raw_message", "")

        if not full_content.strip():
            return ToolResult(
                success=True,
                message=(
                    f"消息 #{message_id} — {nickname}({user_id}) {time_str}\n"
                    "消息内容为空（可能已撤回或无法获取）"
                ),
            )

        # 消息类型信息
        msg_type = data.get("message_type", "")
        group_id = data.get("group_id")
        type_info = ""
        if msg_type == "group" and group_id:
            type_info = f" | 群:{group_id}"
        elif msg_type == "private":
            type_info = " | 私聊"

        lines: list[str] = [
            f"消息 #{message_id} 的完整内容：",
            f"· 发送者：{nickname}({user_id})",
            f"· 时间：{time_str}{type_info}",
            f"· 内容：{full_content}",
        ]

        return ToolResult(success=True, message="\n".join(lines))

    # ──────────────────────────────────────────────
    #  view_message_image
    # ──────────────────────────────────────────────

    async def _handle_view_message_image(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """处理 ``view_message_image`` 操作。

        通过 get_msg API 获取消息详情，提取所有图片 URL，
        根据 enable_vision 决定是否附带图片。

        Args:
            arguments: 工具调用参数（需要 message_id）。
            context: 工具执行上下文。

        Returns:
            包含图片元信息和可选图片附件的结果。
        """
        message_id = arguments.get("message_id")
        if message_id is None:
            return ToolResult(
                success=False,
                message="查看消息图片需要提供 message_id 参数",
            )

        # 获取消息详情
        data = await self._fetch_message(int(message_id), context)
        if data is None:
            return ToolResult(
                success=False,
                message=f"获取消息失败: 消息不存在或已过期 (message_id={message_id})",
            )

        # 解析消息段，提取图片 URL 和元信息
        raw_message_data = data.get("message", [])
        if not isinstance(raw_message_data, list) or not raw_message_data:
            return ToolResult(
                success=True,
                message=f"消息 #{message_id} 不包含图片",
            )

        segments = parse_message(raw_message_data)
        image_urls: list[str] = []
        image_infos: list[str] = []

        for seg in segments:
            if seg.type != "image":
                continue
            url = seg.data.get("url")
            if url:
                image_urls.append(url)

            # 收集元信息
            sub_type = seg.data.get("sub_type")
            label = "自定义表情" if sub_type == 1 else "图片"
            info_parts = [label]
            name = seg.data.get("name", "")
            summary = seg.data.get("summary", "")
            file_hash = seg.data.get("file", "")
            if name:
                info_parts.append(f'name:"{name}"')
            if summary and summary not in ("[图片]", ""):
                info_parts.append(f'summary:"{summary}"')
            if file_hash:
                info_parts.append(f"file:{file_hash}")
            image_infos.append("[" + " ".join(info_parts) + "]")

        if not image_urls:
            return ToolResult(
                success=True,
                message=f"消息 #{message_id} 不包含图片",
            )

        # 构建文本结果
        lines: list[str] = [
            f"消息 #{message_id} 中找到 {len(image_urls)} 张图片：",
        ]
        for i, info in enumerate(image_infos, 1):
            lines.append(f"  {i}. {info}")

        if context.enable_vision:
            lines.append("")
            lines.append("图片已附带，请查看。")
            return ToolResult(
                success=True,
                message="\n".join(lines),
                image_urls=image_urls,
            )
        else:
            lines.append("")
            lines.append("图片识别未启用，无法查看图片内容，仅提供图片元信息。")
            return ToolResult(
                success=True,
                message="\n".join(lines),
            )

    # ──────────────────────────────────────────────
    #  view_avatar
    # ──────────────────────────────────────────────

    async def _handle_view_avatar(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """处理 ``view_avatar`` 操作。

        根据 avatar_type 构建头像 URL，根据 enable_vision 决定是否附带图片。

        Args:
            arguments: 工具调用参数（需要 avatar_type 和 target_id）。
            context: 工具执行上下文。

        Returns:
            包含头像信息和可选图片附件的结果。
        """
        avatar_type = arguments.get("avatar_type")
        target_id = arguments.get("target_id")

        if avatar_type is None:
            return ToolResult(
                success=False,
                message="查看头像需要提供 avatar_type 参数（\"user\" 或 \"group\"）",
            )
        if target_id is None:
            return ToolResult(
                success=False,
                message="查看头像需要提供 target_id 参数（QQ号或群号）",
            )
        if avatar_type not in ("user", "group"):
            return ToolResult(
                success=False,
                message=f"未知的 avatar_type: {avatar_type}，应为 \"user\" 或 \"group\"",
            )

        target_id = int(target_id)

        # 构建头像 URL
        if avatar_type == "user":
            avatar_url = _USER_AVATAR_URL.format(user_id=target_id)
            desc = f"QQ用户 {target_id} 的头像"
        else:
            avatar_url = _GROUP_AVATAR_URL.format(group_id=target_id)
            desc = f"群 {target_id} 的头像"

        if context.enable_vision:
            return ToolResult(
                success=True,
                message=f"{desc}图片已附带，请查看。",
                image_urls=[avatar_url],
            )
        else:
            return ToolResult(
                success=True,
                message=(
                    f"{desc}\n"
                    f"头像 URL：{avatar_url}\n"
                    "图片识别未启用，无法查看头像图片。"
                ),
            )

    # ──────────────────────────────────────────────
    #  内部辅助方法
    # ──────────────────────────────────────────────

    @staticmethod
    async def _fetch_message(
        message_id: int, context: ToolContext
    ) -> dict[str, Any] | None:
        """通过 get_msg API 获取消息详情。

        Args:
            message_id: 消息 ID。
            context: 工具执行上下文。

        Returns:
            消息数据字典，失败时返回 None。
        """
        try:
            data = await context.ws_server.call_api(
                "get_msg",
                {"message_id": message_id},
                timeout=5.0,
            )
        except Exception as e:
            logger.warning(f"获取消息失败 (message_id={message_id}): {e}")
            return None

        if not data or not isinstance(data, dict):
            return None

        return data
