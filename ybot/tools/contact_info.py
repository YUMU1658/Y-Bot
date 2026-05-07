"""联系人查询工具实现。

支持查询好友列表、群列表和用户详细资料（含点赞信息），
与 group_info 工具互补——本工具专注于账号级别的联系人查询。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from ybot.tools.base import BaseTool, ToolContext, ToolResult
from ybot.utils.logger import get_logger

logger = get_logger("联系人工具")

# ── 常量 ──
_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200
# 东八区时区
_TZ_CST = timezone(timedelta(hours=8))


def _format_timestamp(ts: int) -> str:
    """将 Unix 时间戳格式化为可读时间。"""
    if not ts:
        return "未知"
    try:
        dt = datetime.fromtimestamp(ts, tz=_TZ_CST)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError):
        return "未知"


def _sex_label(raw: str) -> str:
    """性别字段转中文。"""
    return {"male": "男", "female": "女"}.get(raw, "未知")


class ContactInfoTool(BaseTool):
    """联系人查询工具。

    通过 OneBot API 查询好友列表、群列表和用户详细资料。
    支持分页（limit/offset）。
    """

    @property
    def name(self) -> str:
        return "contact_info"

    @property
    def description(self) -> str:
        return (
            "查询账号级别的联系人信息。支持三种操作：\n"
            '1. "friend_list" — 获取好友列表，按分组展示（昵称、QQ号、备注、分组）。'
            "默认返回前50个好友，可通过 limit/offset 分页\n"
            '2. "group_list" — 获取已加入的群列表（群名、群号、成员数），'
            "按成员数降序排列，可通过 limit/offset 分页\n"
            '3. "user_profile" — 查询指定用户的详细资料'
            "（昵称、性别、年龄、QQ等级、个性签名、注册时间、VIP、点赞数等），"
            "对好友和非好友均可使用"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["friend_list", "group_list", "user_profile"],
                    "description": "要执行的查询操作",
                },
                "user_id": {
                    "type": "integer",
                    "description": (
                        "要查询详细资料的用户QQ号"
                        "（仅 user_profile 操作需要）"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "friend_list/group_list 操作：返回的最大数量，"
                        "默认50，最大200"
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "friend_list/group_list 操作：从第几个开始（0-based），"
                        "用于分页。默认0"
                    ),
                },
            },
            "required": ["action"],
        }

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """执行联系人查询。

        Args:
            arguments: 包含 ``action`` 及相关参数的字典。
            context: 工具执行上下文。

        Returns:
            查询结果。
        """
        action = arguments.get("action", "")
        if action not in ("friend_list", "group_list", "user_profile"):
            return ToolResult(
                success=False,
                message=f"未知操作: {action}",
            )

        if action == "friend_list":
            return await self._handle_friend_list(arguments, context)
        if action == "group_list":
            return await self._handle_group_list(arguments, context)
        # action == "user_profile"
        user_id = arguments.get("user_id")
        if user_id is None:
            return ToolResult(
                success=False,
                message="查询用户资料需要提供 user_id 参数",
            )
        return await self._handle_user_profile(int(user_id), context)

    # ──────────────────────────────────────────────
    #  friend_list
    # ──────────────────────────────────────────────

    async def _handle_friend_list(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """处理 ``friend_list`` 操作。

        获取完整好友列表后，按分组名称分组展示，支持分页。

        Args:
            arguments: 工具调用参数（可能包含 limit/offset）。
            context: 工具执行上下文。

        Returns:
            格式化的好友列表结果。
        """
        # 1. 获取完整好友列表
        try:
            data = await context.ws_server.call_api(
                "get_friend_list",
                {},
                timeout=10.0,
            )
        except Exception as e:
            logger.warning(f"获取好友列表失败: {e}")
            return ToolResult(
                success=False,
                message=f"获取好友列表失败: {e}",
            )

        # call_api 失败时可能返回 {} 而非 list
        if not isinstance(data, list):
            return ToolResult(
                success=False,
                message="获取好友列表失败: API 返回数据格式异常",
            )

        total_count = len(data)

        # 2. 分页参数
        offset = max(0, int(arguments.get("offset", 0)))
        limit = min(
            max(1, int(arguments.get("limit", _DEFAULT_LIMIT))),
            _MAX_LIMIT,
        )

        # offset 超出范围
        if offset >= total_count and total_count > 0:
            return ToolResult(
                success=True,
                message=(
                    f"offset({offset}) 超出范围，"
                    f"共 {total_count} 名好友"
                ),
            )

        # 3. 按分组名称分组
        category_map: dict[str, list[dict[str, Any]]] = {}
        for friend in data:
            cat_name = friend.get("categoryName") or "我的好友"
            category_map.setdefault(cat_name, []).append(friend)

        # 4. 扁平化后分页截取
        # 保持分组顺序：按分组名称排序，"我的好友"优先
        sorted_categories = sorted(
            category_map.keys(),
            key=lambda c: (0 if c == "我的好友" else 1, c),
        )

        # 构建扁平列表（带分组标记）
        flat_list: list[tuple[str, dict[str, Any]]] = []
        for cat_name in sorted_categories:
            for friend in category_map[cat_name]:
                flat_list.append((cat_name, friend))

        display_items = flat_list[offset: offset + limit]

        # 5. 格式化输出
        if not display_items:
            return ToolResult(
                success=True,
                message="好友列表为空",
            )

        lines: list[str] = []
        display_end = offset + len(display_items)
        lines.append(
            f"好友列表（共 {total_count} 人），"
            f"当前显示第 {offset + 1}~{display_end} 名："
        )

        # 按分组输出（只显示当前页涉及的分组）
        current_category = ""
        category_counts: dict[str, int] = {}
        for cat_name, _ in display_items:
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        for cat_name, friend in display_items:
            if cat_name != current_category:
                current_category = cat_name
                # 显示该分组在完整列表中的总人数
                full_count = len(category_map.get(cat_name, []))
                lines.append("")
                lines.append(f"📁 {cat_name} ({full_count}人)：")

            uid = friend.get("user_id", 0)
            nickname = friend.get("nickname", "")
            remark = friend.get("remark", "")

            entry = f"{nickname}({uid})"
            if remark:
                entry += f" 备注：{remark}"
            lines.append(f"  · {entry}")

        # 分页提示
        remaining = total_count - display_end
        if remaining > 0:
            lines.append("")
            lines.append(
                f"提示：还有 {remaining} 名好友未显示。"
                "可使用 offset 和 limit 参数查看更多。"
            )

        return ToolResult(success=True, message="\n".join(lines))

    # ──────────────────────────────────────────────
    #  group_list
    # ──────────────────────────────────────────────

    async def _handle_group_list(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """处理 ``group_list`` 操作。

        获取完整群列表后，按成员数降序排列，支持分页。

        Args:
            arguments: 工具调用参数（可能包含 limit/offset）。
            context: 工具执行上下文。

        Returns:
            格式化的群列表结果。
        """
        # 1. 获取完整群列表
        try:
            data = await context.ws_server.call_api(
                "get_group_list",
                {},
                timeout=10.0,
            )
        except Exception as e:
            logger.warning(f"获取群列表失败: {e}")
            return ToolResult(
                success=False,
                message=f"获取群列表失败: {e}",
            )

        # call_api 失败时可能返回 {} 而非 list
        if not isinstance(data, list):
            return ToolResult(
                success=False,
                message="获取群列表失败: API 返回数据格式异常",
            )

        total_count = len(data)

        # 2. 按成员数降序排列
        groups = sorted(
            data,
            key=lambda g: g.get("member_count", 0),
            reverse=True,
        )

        # 3. 分页参数
        offset = max(0, int(arguments.get("offset", 0)))
        limit = min(
            max(1, int(arguments.get("limit", _DEFAULT_LIMIT))),
            _MAX_LIMIT,
        )

        # offset 超出范围
        if offset >= total_count and total_count > 0:
            return ToolResult(
                success=True,
                message=(
                    f"offset({offset}) 超出范围，"
                    f"共加入 {total_count} 个群"
                ),
            )

        display_groups = groups[offset: offset + limit]

        # 4. 格式化输出
        if not display_groups:
            return ToolResult(
                success=True,
                message="群列表为空",
            )

        lines: list[str] = []
        display_end = offset + len(display_groups)
        lines.append(
            f"已加入的群聊（共 {total_count} 个），"
            f"当前显示第 {offset + 1}~{display_end} 个："
        )

        for g in display_groups:
            gid = g.get("group_id", 0)
            gname = g.get("group_name", "")
            member_count = g.get("member_count", 0)
            max_member = g.get("max_member_count", 0)
            remark = g.get("group_remark", "")
            all_shut = g.get("group_all_shut", False)

            entry = f"{gname}({gid}) 成员：{member_count}/{max_member}"
            if remark:
                entry += f" 备注：{remark}"
            if all_shut:
                entry += " [全员禁言]"
            lines.append(f"  · {entry}")

        # 分页提示
        remaining = total_count - display_end
        if remaining > 0:
            lines.append("")
            lines.append(
                f"提示：还有 {remaining} 个群未显示。"
                "可使用 offset 和 limit 参数查看更多。"
            )

        return ToolResult(success=True, message="\n".join(lines))

    # ──────────────────────────────────────────────
    #  user_profile
    # ──────────────────────────────────────────────

    async def _handle_user_profile(
        self,
        user_id: int,
        context: ToolContext,
    ) -> ToolResult:
        """处理 ``user_profile`` 操作。

        并发调用 ``get_stranger_info`` 和 ``get_profile_like``，
        整合所有数据格式化输出。``get_profile_like`` 失败时静默跳过。

        Args:
            user_id: 用户 QQ 号。
            context: 工具执行上下文。

        Returns:
            格式化的用户资料信息。
        """
        # 并发调用三个数据源
        stranger_task = context.ws_server.call_api(
            "get_stranger_info",
            {"user_id": user_id, "no_cache": True},
            timeout=5.0,
        )
        profile_like_task = context.ws_server.call_api(
            "get_profile_like",
            {"user_id": str(user_id), "start": 0, "count": 1},
            timeout=5.0,
        )
        friend_task = context.bot_info.is_friend(user_id)

        results = await asyncio.gather(
            stranger_task, profile_like_task, friend_task,
            return_exceptions=True,
        )

        stranger_data = results[0]
        profile_like_data = results[1]
        is_friend = results[2]

        # get_stranger_info 是必须成功的
        if isinstance(stranger_data, Exception):
            logger.warning(
                f"获取用户资料失败 (user_id={user_id}): {stranger_data}"
            )
            return ToolResult(
                success=False,
                message=f"获取用户资料失败: {stranger_data}",
            )

        if not stranger_data or not isinstance(stranger_data, dict):
            return ToolResult(
                success=False,
                message="获取用户资料失败: API 返回数据为空",
            )

        # 格式化输出
        nickname = stranger_data.get("nickname", "")
        lines: list[str] = [
            f"用户资料 — {nickname}({user_id})：",
        ]

        # 基本信息（始终显示）
        lines.append(f"· 昵称：{nickname}")

        # 性别
        sex = stranger_data.get("sex")
        if sex and sex != "unknown":
            lines.append(f"· 性别：{_sex_label(sex)}")

        # 年龄
        age = stranger_data.get("age")
        if age and age != 0:
            lines.append(f"· 年龄：{age}")

        # QQ等级
        qq_level = stranger_data.get("qqLevel")
        if qq_level:
            lines.append(f"· QQ等级：{qq_level}")

        # QID
        qid = stranger_data.get("qid")
        if qid:
            lines.append(f"· QID：{qid}")

        # 个性签名
        long_nick = stranger_data.get("long_nick")
        if long_nick:
            lines.append(f"· 个性签名：{long_nick}")

        # 注册时间
        reg_time = stranger_data.get("reg_time")
        if reg_time:
            lines.append(f"· 注册时间：{_format_timestamp(reg_time)}")

        # 登录天数
        login_days = stranger_data.get("login_days")
        if login_days:
            lines.append(f"· 登录天数：{login_days}")

        # VIP 信息
        is_vip = stranger_data.get("is_vip")
        if is_vip:
            is_years_vip = stranger_data.get("is_years_vip", False)
            vip_level = stranger_data.get("vip_level", 0)
            vip_text = "年费VIP" if is_years_vip else "VIP"
            if vip_level:
                vip_text += f" Lv.{vip_level}"
            lines.append(f"· VIP：{vip_text}")

        # 好友状态
        if isinstance(is_friend, bool):
            lines.append(f"· 是否为Bot好友：{'是' if is_friend else '否'}")

        # 备注（好友才有）
        remark = stranger_data.get("remark")
        if remark:
            lines.append(f"· 备注：{remark}")

        # 点赞信息（get_profile_like 可能失败，静默跳过）
        if (
            not isinstance(profile_like_data, Exception)
            and isinstance(profile_like_data, dict)
        ):
            like_info = self._extract_like_info(profile_like_data)
            if like_info:
                lines.append(f"· 资料点赞：{like_info}")

        return ToolResult(success=True, message="\n".join(lines))

    @staticmethod
    def _extract_like_info(data: dict[str, Any]) -> str:
        """从 get_profile_like 返回数据中提取点赞信息。

        Args:
            data: get_profile_like API 返回的数据。

        Returns:
            格式化的点赞信息字符串，无有效数据时返回空字符串。
        """
        parts: list[str] = []

        # 尝试从 favoriteInfo 提取
        fav_info = data.get("favoriteInfo")
        if isinstance(fav_info, dict):
            total = fav_info.get("total_count")
            today = fav_info.get("today_count")
            if total is not None:
                parts.append(f"总计 {total} 个赞")
            if today is not None:
                parts.append(f"今日 {today} 个赞")

        # 尝试从 voteInfo 提取（备选字段名）
        vote_info = data.get("voteInfo")
        if isinstance(vote_info, dict) and not parts:
            total = vote_info.get("total_count")
            new = vote_info.get("new_count")
            if total is not None:
                parts.append(f"总计 {total} 个赞")
            if new is not None:
                parts.append(f"新增 {new} 个赞")

        return "，".join(parts)
