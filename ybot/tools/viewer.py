"""查看工具实现。

支持查看消息完整内容（文字和/或图片）、合并转发消息、以及用户/群头像。
与引用消息的 80 字符截断互补——本工具可获取完整内容。
"""

from __future__ import annotations

from typing import Any

from ybot.models.message import MessageSegment, parse_message, segments_to_content
from ybot.tools._common import format_timestamp
from ybot.tools.base import BaseTool, ToolContext, ToolResult
from ybot.utils.logger import get_logger

logger = get_logger("查看工具")

# 用户头像 URL 模板
_USER_AVATAR_URL = "https://q.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
# 群头像 URL 模板
_GROUP_AVATAR_URL = "https://p.qlogo.cn/gh/{group_id}/{group_id}/640/"

# view_message 的 include 参数合法值
_VALID_INCLUDE = {"text", "image"}


class ViewerTool(BaseTool):
    """查看工具。

    通过 OneBot API 查看消息完整内容（文字和/或图片）、合并转发消息、
    以及用户/群头像。
    支持三种操作：view_message、view_avatar、view_forward。
    """

    _FORWARD_CACHE_MAX_SIZE = 200  # 最多缓存 200 个嵌套转发 ID

    def __init__(self) -> None:
        # forward_id → {"messages": [...]} 的缓存
        # 用于存储从外层转发消息中提取的嵌套转发内容
        self._forward_cache: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "viewer"

    @property
    def description(self) -> str:
        return (
            "查看消息内容或头像图片。支持三种操作：\n"
            '1. "view_message" — 通过消息ID获取消息的完整内容，'
            "默认同时返回文字和图片。可通过 include 参数控制返回内容："
            'include=["text"] 仅文字，include=["image"] 仅图片，'
            '默认 ["text", "image"] 图文都返回。'
            "需要参数：message_id\n"
            '2. "view_avatar" — 查看QQ用户或群的头像图片，'
            "可查看好友/群友/陌生人的头像。"
            '需要参数：avatar_type（"user"或"group"）、target_id（QQ号或群号）\n'
            '3. "view_forward" — 查看合并转发消息的内容，'
            "通过转发消息ID获取子消息。"
            "默认返回前30条消息，可通过 start（起始位置，从1开始，负数表示从末尾倒数）"
            "和 limit（最多条数）参数控制范围。"
            "需要参数：forward_id"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["view_message", "view_avatar", "view_forward"],
                    "description": "要执行的操作",
                },
                "message_id": {
                    "type": "integer",
                    "description": "要查看的消息ID（view_message 操作需要）",
                },
                "forward_id": {
                    "type": "string",
                    "description": "转发消息ID（仅 view_forward 操作需要）",
                },
                "include": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["text", "image"],
                    },
                    "description": (
                        "要查看的内容类型，默认同时返回文字和图片。"
                        "至少选一个。（view_message / view_forward 操作可用）"
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
                "start": {
                    "type": "integer",
                    "description": (
                        "起始消息位置（从1开始，默认1；"
                        "负数表示从末尾倒数，如-50表示最后50条开始）"
                        "（仅 view_forward 操作可用）"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "最多返回的消息条数（默认30）"
                        "（仅 view_forward 操作可用）"
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

        # 向后兼容：旧的 view_message_image 映射到 view_message + include=["image"]
        if action == "view_message_image":
            arguments.setdefault("include", ["image"])
            action = "view_message"

        if action not in ("view_message", "view_avatar", "view_forward"):
            return ToolResult(
                success=False,
                message=f"未知操作: {action}",
            )

        if action == "view_message":
            return await self._handle_view_message(arguments, context)
        if action == "view_forward":
            return await self._handle_view_forward(arguments, context)
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

        通过 get_msg API 获取消息详情，根据 ``include`` 参数返回文字和/或图片。
        默认同时返回文字和图片内容。

        Args:
            arguments: 工具调用参数（需要 message_id，可选 include）。
            context: 工具执行上下文。

        Returns:
            包含消息内容（文字和/或图片）的结果。
        """
        message_id = arguments.get("message_id")
        if message_id is None:
            return ToolResult(
                success=False,
                message="查看消息需要提供 message_id 参数",
            )

        # 解析 include 参数，默认图文都返回
        raw_include = arguments.get("include")
        if raw_include is None:
            include = _VALID_INCLUDE.copy()
        else:
            include = set(raw_include) & _VALID_INCLUDE
            if not include:
                return ToolResult(
                    success=False,
                    message=(
                        "include 参数无效，"
                        "必须包含 \"text\" 和/或 \"image\""
                    ),
                )

        want_text = "text" in include
        want_image = "image" in include

        # 获取消息详情（只调用一次 API）
        data = await self._fetch_message(int(message_id), context)
        if data is None:
            return ToolResult(
                success=False,
                message=f"获取消息失败: 消息不存在或已过期 (message_id={message_id})",
            )

        # 解析消息段（只解析一次，文字和图片共用）
        raw_message_data = data.get("message", [])
        segments: list[MessageSegment] | None = None
        if isinstance(raw_message_data, list) and raw_message_data:
            segments = parse_message(raw_message_data)

        lines: list[str] = []
        image_urls: list[str] | None = None

        # ── 文字部分 ──
        if want_text:
            sender = data.get("sender", {})
            nickname = sender.get("nickname", "?")
            user_id = data.get("user_id", "?")
            time_str = format_timestamp(data.get("time", 0))

            if segments is not None:
                full_content = segments_to_content(segments)
            else:
                full_content = data.get("raw_message", "")

            msg_type = data.get("message_type", "")
            group_id = data.get("group_id")
            type_info = ""
            if msg_type == "group" and group_id:
                type_info = f" | 群:{group_id}"
            elif msg_type == "private":
                type_info = " | 私聊"

            lines.append(f"消息 #{message_id} 的完整内容：")
            lines.append(f"· 发送者：{nickname}({user_id})")
            lines.append(f"· 时间：{time_str}{type_info}")
            if full_content.strip():
                lines.append(f"· 内容：{full_content}")
            else:
                lines.append("· 内容：（空，可能已撤回或无法获取）")

        # ── 图片部分 ──
        if want_image and segments is not None:
            img_urls, img_infos = self._extract_images(segments)

            if img_urls:
                lines.append(
                    f"消息中找到 {len(img_urls)} 张图片："
                )
                for i, info in enumerate(img_infos, 1):
                    lines.append(f"  {i}. {info}")

                if context.enable_vision:
                    lines.append("图片已附带，请查看。")
                    image_urls = img_urls
                else:
                    lines.append(
                        "图片识别未启用，无法查看图片内容，仅提供图片元信息。"
                    )
            elif not want_text:
                # 只查图片但消息中没有图片
                lines.append(f"消息 #{message_id} 不包含图片")
        elif want_image and segments is None and not want_text:
            # 只查图片但消息段为空
            lines.append(f"消息 #{message_id} 不包含图片")

        return ToolResult(
            success=True,
            message="\n".join(lines),
            image_urls=image_urls,
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
    #  view_forward
    # ──────────────────────────────────────────────

    async def _handle_view_forward(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """处理 ``view_forward`` 操作。

        通过 get_forward_msg API 获取合并转发消息的完整内容，
        根据 ``include`` 参数返回文字和/或图片。

        Args:
            arguments: 工具调用参数（需要 forward_id，可选 include）。
            context: 工具执行上下文。

        Returns:
            包含转发消息内容（文字和/或图片）的结果。
        """
        forward_id = arguments.get("forward_id")
        if not forward_id:
            return ToolResult(
                success=False,
                message="查看转发消息需要提供 forward_id 参数",
            )

        # 解析 include 参数，默认图文都返回
        raw_include = arguments.get("include")
        if raw_include is None:
            include = _VALID_INCLUDE.copy()
        else:
            include = set(raw_include) & _VALID_INCLUDE
            if not include:
                return ToolResult(
                    success=False,
                    message=(
                        "include 参数无效，"
                        "必须包含 \"text\" 和/或 \"image\""
                    ),
                )

        want_text = "text" in include
        want_image = "image" in include

        # 获取转发消息
        data = await self._fetch_forward(str(forward_id), context)
        if data is None:
            return ToolResult(
                success=False,
                message=(
                    "获取转发消息失败: "
                    f"转发消息不存在或已过期 (forward_id={forward_id})"
                ),
            )

        messages = data.get("messages", [])
        if not messages:
            return ToolResult(
                success=False,
                message=f"转发消息内容为空 (forward_id={forward_id})",
            )

        total = len(messages)

        # 解析范围参数
        start = arguments.get("start", 1)
        limit = arguments.get("limit", 30)

        # 防御性处理
        if not isinstance(start, int):
            start = 1
        if not isinstance(limit, int) or limit <= 0:
            limit = 30
        if start == 0:
            start = 1

        # 处理负数 start（从末尾倒数）
        if start < 0:
            start_idx = max(0, total + start)
        else:
            start_idx = max(0, start - 1)  # 转换为 0-based

        end_idx = min(total, start_idx + limit)

        # 切片
        display_messages = messages[start_idx:end_idx]
        display_start = start_idx + 1  # 1-based 起始
        display_end = end_idx  # 1-based 结束

        # 构建头部
        if display_start == 1 and display_end == total:
            # 显示全部消息
            header_line = (
                f"转发消息 #{forward_id} 的完整内容（共{total}条）："
            )
        else:
            header_line = (
                f"转发消息 #{forward_id} 的内容"
                f"（共{total}条，当前显示第{display_start}-{display_end}条）："
            )

        lines: list[str] = [header_line]
        all_image_urls: list[str] = []
        # (node_index, url, info_str) 三元组
        all_image_infos: list[tuple[int, str, str]] = []

        for i, node in enumerate(display_messages):
            idx = start_idx + i + 1  # 原始 1-based 索引
            nickname, user_id, raw_segs = self._extract_forward_node(node)

            # 提取时间戳（兼容 NapCat 扁平格式和 go-cqhttp 包装格式）
            timestamp = node.get("time", 0)
            if not timestamp:
                timestamp = (node.get("data") or {}).get("time", 0)

            # 构建发送者显示
            if nickname and user_id:
                sender = f"{nickname}({user_id})"
            elif nickname:
                sender = nickname
            elif user_id:
                sender = user_id
            else:
                sender = "?"

            # 构建头部行
            time_str = format_timestamp(timestamp) if timestamp else ""
            header = f"[{idx}] {sender}"
            if time_str and time_str != "未知":
                header += f" {time_str}"

            # 解析消息段
            segments = (
                parse_message(raw_segs)
                if raw_segs and isinstance(raw_segs, list)
                else []
            )

            # ── 文字部分 ──
            if want_text:
                content = (
                    segments_to_content(segments) if segments else ""
                )
                if not content.strip():
                    # 空消息：将提示追加到头部元信息行，防止伪造
                    lines.append(f"\n{header} （无内容）")
                else:
                    lines.append(f"\n{header}")
                    lines.append(content)

            # ── 图片部分 ──
            if want_image and segments:
                img_urls, img_infos = self._extract_images(segments)
                all_image_urls.extend(img_urls)
                all_image_infos.extend(
                    (idx, url, info)
                    for url, info in zip(img_urls, img_infos)
                )

        # 图片汇总
        image_urls: list[str] | None = None
        if want_image and all_image_urls:
            lines.append(
                f"\n消息中共找到 {len(all_image_urls)} 张图片："
            )
            for node_idx, _url, info in all_image_infos:
                lines.append(f"  消息[{node_idx}]: {info}")

            if context.enable_vision:
                lines.append("图片已附带，请查看。")
                image_urls = [url for _, url, _ in all_image_infos]
            else:
                lines.append(
                    "图片识别未启用，无法查看图片内容，仅提供图片元信息。"
                )

        # 截断提示
        remaining = total - display_end
        if remaining > 0:
            lines.append(
                f"\n（还有{remaining}条消息未显示，"
                "可通过 start/limit 参数查看更多）"
            )

        return ToolResult(
            success=True,
            message="\n".join(lines),
            image_urls=image_urls,
        )

    # ──────────────────────────────────────────────
    #  内部辅助方法
    # ──────────────────────────────────────────────

    @staticmethod
    def _extract_images(
        segments: list[MessageSegment],
    ) -> tuple[list[str], list[str]]:
        """从消息段列表中提取所有图片 URL 和元信息。

        Args:
            segments: 已解析的消息段列表。

        Returns:
            ``(image_urls, image_infos)`` 元组：
            - image_urls: 图片 URL 列表。
            - image_infos: 对应的元信息描述列表。
        """
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

        return image_urls, image_infos

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

    @staticmethod
    def _extract_forward_node(
        node: dict[str, Any],
    ) -> tuple[str, str, list[Any]]:
        """从转发消息 node 中提取发送者信息和消息段。

        兼容两种格式：

        - NapCat 格式：扁平结构（sender.nickname, user_id, message）
        - go-cqhttp 格式：node 包装（data.nickname, data.user_id, data.message）

        与 ``bot.py`` 的 ``_extract_node_info`` 不同，此方法 **不** 读取
        ``sender.card``（群名片），因为合并转发消息中的群名片不可靠。

        Args:
            node: 转发消息中的单条 node 字典。

        Returns:
            ``(nickname, user_id, message_segments)`` 三元组。
            任何字段缺失时返回空字符串/空列表。
        """
        # NapCat 扁平格式
        sender = node.get("sender")
        if isinstance(sender, dict) and sender:
            nickname = sender.get("nickname") or ""
            user_id = (
                str(node.get("user_id", ""))
                if node.get("user_id")
                else ""
            )
            raw_segs = node.get("message", [])
            if nickname or user_id or raw_segs:
                return (
                    nickname,
                    user_id,
                    raw_segs if isinstance(raw_segs, list) else [],
                )

        # go-cqhttp node 包装格式
        node_data = (
            node.get("data", {}) if isinstance(node, dict) else {}
        )
        nickname = node_data.get("nickname", "")
        user_id = (
            str(node_data.get("user_id", ""))
            if node_data.get("user_id")
            else ""
        )
        raw_segs = node_data.get("message", [])
        return (
            nickname,
            user_id,
            raw_segs if isinstance(raw_segs, list) else [],
        )

    def _cache_nested_forwards(
        self,
        messages: list[dict[str, Any]],
        *,
        depth: int = 0,
        max_depth: int = 5,
    ) -> None:
        """递归扫描转发消息中的嵌套 forward segment，缓存其内联内容。

        NapCat 的 ``get_forward_msg`` 会将嵌套转发的完整内容内联在
        forward segment 的 ``data.content`` 中。本方法将这些内容提取出来，
        以嵌套 forward 的 ``data.id`` 为 key 缓存到 ``_forward_cache``，
        供后续 ``_fetch_forward`` 直接返回，避免再次调用 API（NapCat 会报错
        "消息已过期或者为内层消息"）。

        Args:
            messages: get_forward_msg 返回的 messages 数组（node 列表）。
            depth: 当前递归深度。
            max_depth: 最大递归深度，防止恶意构造的深层嵌套。
        """
        if depth >= max_depth:
            return
        if len(self._forward_cache) >= self._FORWARD_CACHE_MAX_SIZE:
            return  # 缓存已满，停止缓存新条目

        for node in messages:
            if not isinstance(node, dict):
                continue

            # 获取消息段列表（兼容 NapCat 扁平格式和 go-cqhttp 包装格式）
            raw_segs = node.get("message", [])
            if not isinstance(raw_segs, list):
                node_data = node.get("data", {})
                if isinstance(node_data, dict):
                    raw_segs = node_data.get("message", [])
                else:
                    raw_segs = []

            for seg in raw_segs:
                if not isinstance(seg, dict):
                    continue
                if seg.get("type") != "forward":
                    continue
                seg_data = seg.get("data", {})
                if not isinstance(seg_data, dict):
                    continue

                nested_id = seg_data.get("id")
                content = seg_data.get("content")

                if nested_id and isinstance(content, list) and content:
                    nested_id_str = str(nested_id)
                    # 将嵌套内容缓存为与 get_forward_msg 相同的格式
                    self._forward_cache[nested_id_str] = {
                        "messages": content,
                    }
                    logger.debug(
                        f"缓存嵌套转发消息 (id={nested_id_str}, "
                        f"depth={depth + 1}, "
                        f"子消息数={len(content)})"
                    )
                    # 递归处理更深层的嵌套
                    self._cache_nested_forwards(
                        content,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )

                    if len(self._forward_cache) >= self._FORWARD_CACHE_MAX_SIZE:
                        return  # 缓存已满，提前退出

    async def _fetch_forward(
        self, forward_id: str, context: ToolContext
    ) -> dict[str, Any] | None:
        """通过缓存或 get_forward_msg API 获取转发消息内容。

        优先查询内部缓存（嵌套转发内容），未命中时调用 API。
        API 返回后自动扫描并缓存嵌套的转发内容。

        Args:
            forward_id: 转发消息 ID。
            context: 工具执行上下文。

        Returns:
            转发消息数据字典，失败时返回 None。
        """
        # 1. 先查缓存（pop: 用完即删，避免内存泄漏）
        cached = self._forward_cache.pop(forward_id, None)
        if cached is not None:
            logger.debug(f"从缓存获取嵌套转发消息 (id={forward_id})")
            # 缓存的数据也可能包含更深层嵌套，递归缓存
            self._cache_nested_forwards(cached.get("messages", []))
            return cached

        # 2. 缓存未命中，调用 API
        try:
            data = await context.ws_server.call_api(
                "get_forward_msg",
                {"id": forward_id},
                timeout=10.0,
            )
        except Exception as e:
            logger.warning(f"获取转发消息失败 (id={forward_id}): {e}")
            return None

        if not data or not isinstance(data, dict):
            return None

        # 3. 扫描并缓存嵌套的转发内容
        messages = data.get("messages", [])
        if messages:
            self._cache_nested_forwards(messages)

        return data
