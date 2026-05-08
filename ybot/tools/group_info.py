"""群信息查询工具实现。

支持查询群成员列表、个人详细信息和群资料，
包含灵活的分页、角色筛选和自动群号推断。
"""

from __future__ import annotations

from typing import Any

from ybot.constants import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, ROLE_LABEL_MAP
from ybot.tools._common import format_timestamp
from ybot.tools.base import BaseTool, ToolContext, ToolResult
from ybot.utils.logger import get_logger

logger = get_logger("群信息工具")

# ── 常量 ──
_ROLE_PRIORITY: dict[str, int] = {"owner": 0, "admin": 1, "member": 2}


def _normalize_level(raw: Any) -> str:
    """将 API 返回的 level 字段规范化为字符串。

    - 0 或空值 → 空字符串（无等级）
    - 正整数 → 数字字符串（如 ``"8"``）
    - 非空字符串 → 原样保留（如 ``"冒泡达人"``）
    """
    if raw is None or raw == "" or raw == 0:
        return ""
    return str(raw)


def _level_sort_value(raw: Any) -> int:
    """提取等级的数值用于排序，无法解析时返回 0。"""
    try:
        return int(raw or 0)
    except (ValueError, TypeError):
        return 0


def _format_level(level_str: str) -> str:
    """格式化等级显示。

    纯数字 → ``Lv.8``，含中文名称 → ``Lv.0「冒泡达人」``，空 → 空字符串。
    """
    if not level_str:
        return ""
    try:
        num = int(level_str)
        return f"Lv.{num}"
    except ValueError:
        # 非数字等级名称（如 "冒泡达人"）
        return f"「{level_str}」"


def _normalize_role(raw: Any) -> str:
    """规范化角色字段，空/缺失视为 ``member``。"""
    role = str(raw) if raw else ""
    if role not in _ROLE_PRIORITY:
        return "member"
    return role


def _sex_label(raw: str) -> str:
    """性别字段转中文。"""
    return {"male": "男", "female": "女"}.get(raw, "未知")


class GroupInfoTool(BaseTool):
    """群信息查询工具。

    通过 OneBot API 查询群成员列表、个人详细信息和群资料。
    支持灵活的分页（limit/offset）和角色筛选（role_filter）。
    """

    @property
    def name(self) -> str:
        return "group_info"

    @property
    def description(self) -> str:
        return (
            "查询群聊相关信息。支持三种操作：\n"
            '1. "member_list" — 获取群成员列表（昵称、群名片、角色、等级、头衔）。'
            "默认返回前50个成员，可通过 limit/offset 控制范围，"
            "通过 role_filter 筛选角色\n"
            '2. "member_detail" — 获取某个成员的详细信息'
            "（个性签名、加入时间、最后发言、年龄等）\n"
            '3. "group_detail" — 获取群资料'
            "（群名、群介绍、成员数、创建时间等）\n"
            "当前在群聊中时 group_id 可省略，将自动使用当前群。"
            "私聊中必须提供 group_id。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["member_list", "member_detail", "group_detail"],
                    "description": "要执行的查询操作",
                },
                "group_id": {
                    "type": "integer",
                    "description": (
                        "目标群号。在群聊中可省略（自动使用当前群），"
                        "私聊中必须提供"
                    ),
                },
                "user_id": {
                    "type": "integer",
                    "description": (
                        "要查询详细信息的成员QQ号"
                        "（仅 member_detail 操作需要）"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "member_list 操作：返回的最大成员数量，"
                        "默认50，最大200"
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "member_list 操作：从第几个成员开始（0-based），"
                        "用于分页/范围查询。默认0"
                    ),
                },
                "role_filter": {
                    "type": "string",
                    "enum": ["owner", "admin", "member", "admin+"],
                    "description": (
                        "member_list 操作：按角色筛选。"
                        "admin+ 表示管理员和群主"
                    ),
                },
            },
            "required": ["action"],
        }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """执行群信息查询。

        Args:
            arguments: 包含 ``action`` 及相关参数的字典。
            context: 工具执行上下文。

        Returns:
            查询结果。
        """
        action = arguments.get("action", "")
        if action not in ("member_list", "member_detail", "group_detail"):
            return ToolResult(
                success=False,
                message=f"未知操作: {action}",
            )

        # 解析 group_id
        group_id = self._resolve_group_id(arguments, context)
        if group_id is None:
            return ToolResult(
                success=False,
                message="当前不在群聊中，请提供 group_id 参数",
            )

        if action == "member_list":
            return await self._handle_member_list(group_id, arguments, context)
        if action == "member_detail":
            user_id = arguments.get("user_id")
            if user_id is None:
                return ToolResult(
                    success=False,
                    message="查询成员详细信息需要提供 user_id 参数",
                )
            return await self._handle_member_detail(
                group_id, int(user_id), context
            )
        # action == "group_detail"
        return await self._handle_group_detail(group_id, context)

    # ──────────────────────────────────────────────
    #  group_id 解析
    # ──────────────────────────────────────────────

    @staticmethod
    def _resolve_group_id(
        arguments: dict[str, Any], context: ToolContext
    ) -> int | None:
        """从参数或 session_key 中解析群号。

        优先使用参数中显式提供的 ``group_id``，否则尝试从
        ``session_key`` 中提取（支持 ``group_`` 和 ``temp_`` 前缀）。

        Returns:
            群号，或 ``None`` 表示无法确定。
        """
        # 优先使用参数中的 group_id
        gid = arguments.get("group_id")
        if gid is not None:
            return int(gid)

        # 从 session_key 中提取
        sk = context.session_key
        if sk.startswith("group_"):
            return int(sk.removeprefix("group_"))

        # 临时会话也可以提取来源群
        if sk.startswith("temp_"):
            parts = sk.split("_")
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass

        return None  # 无法确定群号

    # ──────────────────────────────────────────────
    #  member_list
    # ──────────────────────────────────────────────

    async def _handle_member_list(
        self,
        group_id: int,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """处理 ``member_list`` 操作。

        全量获取群成员列表后，在内存中完成角色筛选、排序和分页截取。

        Args:
            group_id: 群号。
            arguments: 工具调用参数（可能包含 limit/offset/role_filter）。
            context: 工具执行上下文。

        Returns:
            格式化的成员列表结果。
        """
        # 1. 获取全量成员列表
        try:
            data = await context.ws_server.call_api(
                "get_group_member_list",
                {"group_id": group_id},
                timeout=10.0,
            )
        except Exception as e:
            logger.warning(f"获取群成员列表失败 (group_id={group_id}): {e}")
            return ToolResult(
                success=False,
                message=f"获取群成员列表失败: {e}",
            )

        # call_api 失败时可能返回 {} 而非 list
        if not isinstance(data, list):
            return ToolResult(
                success=False,
                message="获取群成员列表失败: API 返回数据格式异常",
            )

        total_count = len(data)

        # 2. 角色筛选
        role_filter: str | None = arguments.get("role_filter")
        if role_filter:
            if role_filter == "admin+":
                members = [
                    m for m in data
                    if _normalize_role(m.get("role")) in ("owner", "admin")
                ]
            else:
                members = [
                    m for m in data
                    if _normalize_role(m.get("role")) == role_filter
                ]
        else:
            members = list(data)

        filtered_count = len(members)

        # 3. 排序：角色优先级 → 等级降序
        def _sort_key(m: dict[str, Any]) -> tuple[int, int]:
            role_order = _ROLE_PRIORITY.get(
                _normalize_role(m.get("role")), 2
            )
            level_val = _level_sort_value(m.get("level"))
            return (role_order, -level_val)

        members.sort(key=_sort_key)

        # 4. 分页截取
        offset = max(0, int(arguments.get("offset", 0)))
        limit = min(
            max(1, int(arguments.get("limit", DEFAULT_PAGE_LIMIT))),
            MAX_PAGE_LIMIT,
        )

        # offset 超出范围
        if offset >= filtered_count and filtered_count > 0:
            filter_hint = (
                f"（{self._role_filter_label(role_filter)}）"
                if role_filter
                else ""
            )
            return ToolResult(
                success=True,
                message=(
                    f"offset({offset}) 超出范围，"
                    f"该群共 {total_count} 名成员"
                    f"{filter_hint}符合条件的有 {filtered_count} 人"
                ),
            )

        display_members = members[offset: offset + limit]

        # 5. 获取群名（用于标题）
        group_name = await self._fetch_group_name(group_id, context)

        # 6. 格式化输出
        return self._format_member_list(
            group_id=group_id,
            group_name=group_name,
            display_members=display_members,
            total_count=total_count,
            filtered_count=filtered_count,
            offset=offset,
            limit=limit,
            role_filter=role_filter,
        )

    def _format_member_list(
        self,
        *,
        group_id: int,
        group_name: str,
        display_members: list[dict[str, Any]],
        total_count: int,
        filtered_count: int,
        offset: int,
        limit: int,
        role_filter: str | None,
    ) -> ToolResult:
        """将成员列表格式化为可读文本。"""
        if not display_members:
            filter_hint = (
                f"（筛选条件: {self._role_filter_label(role_filter)}）"
                if role_filter
                else ""
            )
            return ToolResult(
                success=True,
                message=f"未找到符合条件的成员{filter_hint}",
            )

        lines: list[str] = []

        # 标题行
        display_end = offset + len(display_members)
        if role_filter:
            filter_label = self._role_filter_label(role_filter)
            lines.append(
                f"群「{group_name}」(ID:{group_id}) "
                f"{filter_label}共 {filtered_count} 人"
                f"（群总人数 {total_count}），"
                f"当前显示第 {offset + 1}~{display_end} 名："
            )
        else:
            lines.append(
                f"群「{group_name}」(ID:{group_id}) "
                f"共 {total_count} 名成员，"
                f"当前显示第 {offset + 1}~{display_end} 名"
                f"（按角色和等级排序）："
            )

        # 按角色分组
        owners: list[str] = []
        admins: list[str] = []
        normals: list[str] = []

        for m in display_members:
            entry = self._format_member_entry(m)
            role = _normalize_role(m.get("role"))
            if role == "owner":
                owners.append(entry)
            elif role == "admin":
                admins.append(entry)
            else:
                normals.append(entry)

        if owners:
            lines.append("")
            lines.append("👑 群主：")
            for e in owners:
                lines.append(f"  · {e}")

        if admins:
            lines.append("")
            lines.append(f"🛡️ 管理员 ({len(admins)}人)：")
            for e in admins:
                lines.append(f"  · {e}")

        if normals:
            lines.append("")
            lines.append(f"👤 普通成员 ({len(normals)}人)：")
            for e in normals:
                lines.append(f"  · {e}")

        # 分页提示
        remaining = filtered_count - display_end
        if remaining > 0:
            lines.append("")
            lines.append(
                f"提示：还有 {remaining} 名成员未显示。"
                "可使用 offset 和 limit 参数查看更多，"
                "或使用 role_filter 按角色筛选。"
            )

        return ToolResult(success=True, message="\n".join(lines))

    @staticmethod
    def _format_member_entry(m: dict[str, Any]) -> str:
        """格式化单个成员条目。"""
        uid = m.get("user_id", 0)
        nickname = m.get("nickname", "")
        card = m.get("card", "")
        level = _normalize_level(m.get("level"))
        title = m.get("title", "")

        parts: list[str] = [f"{nickname}({uid})"]

        if card:
            parts.append(f"群名片：{card}")

        level_display = _format_level(level)
        if level_display:
            parts.append(level_display)

        if title:
            parts.append(f"头衔：{title}")

        return " | ".join(parts)

    @staticmethod
    def _role_filter_label(role_filter: str | None) -> str:
        """返回角色筛选条件的中文描述。"""
        if role_filter == "admin+":
            return "管理层（群主+管理员）"
        if role_filter in ROLE_LABEL_MAP:
            return ROLE_LABEL_MAP[role_filter]
        return ""

    # ──────────────────────────────────────────────
    #  member_detail
    # ──────────────────────────────────────────────

    async def _handle_member_detail(
        self,
        group_id: int,
        user_id: int,
        context: ToolContext,
    ) -> ToolResult:
        """处理 ``member_detail`` 操作。

        Args:
            group_id: 群号。
            user_id: 成员 QQ 号。
            context: 工具执行上下文。

        Returns:
            格式化的成员详细信息。
        """
        try:
            data = await context.ws_server.call_api(
                "get_group_member_info",
                {"group_id": group_id, "user_id": user_id, "no_cache": True},
                timeout=5.0,
            )
        except Exception as e:
            logger.warning(
                f"获取成员详细信息失败 "
                f"(group={group_id}, user={user_id}): {e}"
            )
            return ToolResult(
                success=False,
                message=f"获取成员详细信息失败: {e}",
            )

        if not data or not isinstance(data, dict):
            return ToolResult(
                success=False,
                message="获取成员详细信息失败: API 返回数据为空",
            )

        # 获取群名
        group_name = await self._fetch_group_name(group_id, context)

        nickname = data.get("nickname", "")
        lines: list[str] = [
            f"成员详细信息 — {nickname}({user_id}) "
            f"@ {group_name}({group_id})：",
        ]

        # 基本信息（始终显示）
        lines.append(f"· 昵称：{nickname}")

        card = data.get("card", "")
        if card:
            lines.append(f"· 群名片：{card}")

        role = _normalize_role(data.get("role"))
        lines.append(f"· 角色：{ROLE_LABEL_MAP.get(role, role)}")

        # 等级
        level = _normalize_level(data.get("level"))
        level_display = _format_level(level)
        title = data.get("title", "")
        if level_display and title:
            lines.append(f"· 群等级：{level_display}「{title}」")
        elif level_display:
            lines.append(f"· 群等级：{level_display}")
        elif title:
            lines.append(f"· 专属头衔：{title}")

        # 可选字段（取决于 OneBot 实现端）
        sex = data.get("sex")
        if sex and sex != "unknown":
            lines.append(f"· 性别：{_sex_label(sex)}")

        age = data.get("age")
        if age and age != 0:
            lines.append(f"· 年龄：{age}")

        qq_level = data.get("qq_level")
        if qq_level:
            lines.append(f"· QQ等级：{qq_level}")

        sign = data.get("sign")
        if sign:
            lines.append(f"· 个性签名：{sign}")

        join_time = data.get("join_time")
        if join_time:
            lines.append(f"· 入群时间：{format_timestamp(join_time)}")

        last_sent = data.get("last_sent_time")
        if last_sent:
            lines.append(f"· 最后发言：{format_timestamp(last_sent)}")

        # 好友状态
        try:
            is_friend = await context.bot_info.is_friend(user_id)
            lines.append(f"· 是否为Bot好友：{'是' if is_friend else '否'}")
        except Exception:
            pass  # 好友查询失败不影响主要结果

        # 禁言状态
        shut_up = data.get("shut_up_timestamp", 0)
        if shut_up and shut_up > 0:
            lines.append(f"· 禁言到期：{format_timestamp(shut_up)}")

        area = data.get("area")
        if area:
            lines.append(f"· 地区：{area}")

        return ToolResult(success=True, message="\n".join(lines))

    # ──────────────────────────────────────────────
    #  group_detail
    # ──────────────────────────────────────────────

    async def _handle_group_detail(
        self,
        group_id: int,
        context: ToolContext,
    ) -> ToolResult:
        """处理 ``group_detail`` 操作。

        Args:
            group_id: 群号。
            context: 工具执行上下文。

        Returns:
            格式化的群资料信息。
        """
        try:
            data = await context.ws_server.call_api(
                "get_group_detail_info",
                {"group_id": group_id},
                timeout=5.0,
            )
        except Exception as e:
            logger.warning(f"获取群信息失败 (group_id={group_id}): {e}")
            return ToolResult(
                success=False,
                message=f"获取群信息失败: {e}",
            )

        if not data or not isinstance(data, dict):
            return ToolResult(
                success=False,
                message="获取群信息失败: API 返回数据为空",
            )

        group_name = data.get("group_name") or data.get("groupName", "")
        lines: list[str] = [
            f"群资料 — {group_name}(ID:{group_id})：",
        ]

        lines.append(f"· 群名：{group_name}")
        lines.append(f"· 群号：{group_id}")

        member_count = data.get("member_count") or data.get("memberNum")
        if member_count is not None:
            lines.append(f"· 当前成员数：{member_count}")

        max_member = data.get("max_member_count") or data.get("maxMemberNum")
        if max_member is not None:
            lines.append(f"· 最大成员数：{max_member}")

        # fingerMemo 才是群资料卡上的「群介绍」
        # groupMemo 是「群公告」，不在此处显示
        intro = data.get("fingerMemo") or data.get("finger_memo")
        if intro:
            lines.append(f"· 群介绍：{intro}")

        create_time = data.get("groupCreateTime") or data.get("group_create_time")
        if create_time:
            lines.append(f"· 创建时间：{format_timestamp(create_time)}")

        group_level = data.get("group_level") or data.get("groupGrade")
        if group_level:
            lines.append(f"· 群等级：{group_level}")

        return ToolResult(success=True, message="\n".join(lines))

    # ──────────────────────────────────────────────
    #  辅助方法
    # ──────────────────────────────────────────────

    @staticmethod
    async def _fetch_group_name(
        group_id: int, context: ToolContext
    ) -> str:
        """获取群名称，失败时返回空字符串。"""
        try:
            info = await context.bot_info.get_group_info(group_id)
            return info.group_name or ""
        except Exception:
            return ""
