"""Bot 信息缓存服务。

缓存 Bot 自身信息、群信息、群成员信息和好友列表，
通过 OneBot API 获取数据并按 TTL 自动刷新。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ybot.core.ws_server import WebSocketServer
from ybot.utils.logger import get_logger

logger = get_logger("信息缓存")

# 缓存 TTL（秒）
_GROUP_TTL = 300  # 群信息 5 分钟
_MEMBER_TTL = 300  # 群成员信息 5 分钟
_FRIEND_TTL = 600  # 好友列表 10 分钟

# 缓存最大条目数（防止无界增长）
_GROUP_CACHE_MAX = 500
_MEMBER_CACHE_MAX = 5000


def _normalize_level(raw: Any) -> str:
    """将 API 返回的 level 字段规范化为字符串。

    - 0 或空值 → 空字符串（无等级）
    - 正整数 → 数字字符串（如 "8"）
    - 非空字符串 → 原样保留（如 "冒泡达人"）
    """
    if raw is None or raw == "" or raw == 0:
        return ""
    return str(raw)


@dataclass
class GroupInfo:
    """群信息缓存条目。"""

    group_id: int = 0
    group_name: str = ""
    member_count: int = 0
    fetched_at: float = 0.0


@dataclass
class MemberInfo:
    """群成员信息缓存条目。"""

    user_id: int = 0
    nickname: str = ""
    card: str = ""
    role: str = ""  # owner / admin / member
    level: str = ""  # 群等级（可能是数字或等级名称，取决于 OneBot 实现）
    title: str = ""  # 专属头衔（群主授予）
    fetched_at: float = 0.0


@dataclass
class BotLoginInfo:
    """Bot 登录信息。"""

    user_id: int = 0
    nickname: str = ""


class BotInfoService:
    """Bot 信息缓存服务。

    通过 WebSocketServer.call_api() 调用 OneBot API 获取信息，
    并按 TTL 缓存以减少 API 调用频率。

    Attributes:
        _ws: WebSocket 服务端引用。
        _login_info: Bot 登录信息（启动时获取，永久缓存）。
        _group_cache: 群信息缓存，key 为 group_id。
        _member_cache: 群成员信息缓存，key 为 (group_id, user_id)。
        _friend_set: 好友 QQ 号集合。
        _friend_fetched_at: 好友列表上次获取时间。
    """

    def __init__(self, ws_server: WebSocketServer) -> None:
        self._ws = ws_server
        self._login_info: BotLoginInfo = BotLoginInfo()
        self._group_cache: dict[int, GroupInfo] = {}
        self._member_cache: dict[tuple[int, int], MemberInfo] = {}
        self._friend_set: set[int] = set()
        self._friend_fetched_at: float = 0.0

    async def initialize(self) -> None:
        """初始化服务，获取 Bot 登录信息和好友列表。

        应在 WebSocket 客户端连接后调用。如果调用失败（如客户端尚未连接），
        会记录警告但不会阻止启动。
        """
        await self._fetch_login_info()
        await self._fetch_friend_list()

    async def get_login_info(self) -> BotLoginInfo:
        """获取 Bot 登录信息。

        Returns:
            Bot 登录信息（user_id 和 nickname）。
        """
        if self._login_info.user_id == 0:
            await self._fetch_login_info()
        return self._login_info

    async def get_group_info(self, group_id: int) -> GroupInfo:
        """获取群信息（带缓存）。

        Args:
            group_id: 群号。

        Returns:
            群信息。如果 API 调用失败，返回仅含 group_id 的默认值。
        """
        cached = self._group_cache.get(group_id)
        if cached is not None and (time.time() - cached.fetched_at) < _GROUP_TTL:
            return cached

        try:
            data = await self._ws.call_api("get_group_info", {"group_id": group_id})
            info = GroupInfo(
                group_id=group_id,
                group_name=data.get("group_name", ""),
                member_count=data.get("member_count", 0),
                fetched_at=time.time(),
            )
            self._group_cache[group_id] = info
            # 防止缓存无界增长：超过上限时清理最旧的条目
            if len(self._group_cache) > _GROUP_CACHE_MAX:
                self._evict_oldest_groups()
            return info
        except Exception as e:
            logger.warning(f"获取群信息失败 (group_id={group_id}): {e}")
            # 返回缓存中的旧数据（如果有），否则返回默认值
            if cached is not None:
                return cached
            return GroupInfo(group_id=group_id)

    async def get_member_info(self, group_id: int, user_id: int) -> MemberInfo:
        """获取群成员信息（带缓存）。

        Args:
            group_id: 群号。
            user_id: 成员 QQ 号。

        Returns:
            成员信息。如果 API 调用失败，返回仅含 user_id 的默认值。
        """
        key = (group_id, user_id)
        cached = self._member_cache.get(key)
        if cached is not None and (time.time() - cached.fetched_at) < _MEMBER_TTL:
            return cached

        try:
            data = await self._ws.call_api(
                "get_group_member_info",
                {"group_id": group_id, "user_id": user_id},
            )
            info = MemberInfo(
                user_id=user_id,
                nickname=data.get("nickname", ""),
                card=data.get("card", ""),
                role=data.get("role", ""),
                level=_normalize_level(data.get("level")),
                title=data.get("title", ""),
                fetched_at=time.time(),
            )
            self._member_cache[key] = info
            # 防止缓存无界增长：超过上限时清理最旧的条目
            if len(self._member_cache) > _MEMBER_CACHE_MAX:
                self._evict_oldest_members()
            return info
        except Exception as e:
            logger.warning(
                f"获取群成员信息失败 (group={group_id}, user={user_id}): {e}"
            )
            if cached is not None:
                return cached
            return MemberInfo(user_id=user_id)

    async def is_friend(self, user_id: int) -> bool:
        """判断指定用户是否为 Bot 的好友。

        Args:
            user_id: 用户 QQ 号。

        Returns:
            是否为好友。如果好友列表获取失败，返回 False。
        """
        # 检查是否需要刷新好友列表
        if (time.time() - self._friend_fetched_at) >= _FRIEND_TTL:
            await self._fetch_friend_list()
        return user_id in self._friend_set

    async def _fetch_login_info(self) -> None:
        """获取 Bot 登录信息。"""
        try:
            data = await self._ws.call_api("get_login_info", {})
            self._login_info = BotLoginInfo(
                user_id=data.get("user_id", 0),
                nickname=data.get("nickname", ""),
            )
            logger.info(
                f"Bot 登录信息: {self._login_info.nickname}({self._login_info.user_id})"
            )
        except Exception as e:
            logger.warning(f"获取 Bot 登录信息失败: {e}")

    async def _fetch_friend_list(self) -> None:
        """获取好友列表并缓存。"""
        try:
            data = await self._ws.call_api("get_friend_list", {})
            if isinstance(data, list):
                self._friend_set = {
                    int(f.get("user_id", 0)) for f in data if f.get("user_id")
                }
            self._friend_fetched_at = time.time()
            logger.debug(f"好友列表已刷新，共 {len(self._friend_set)} 人")
        except Exception as e:
            logger.warning(f"获取好友列表失败: {e}")

    def _evict_oldest_groups(self) -> None:
        """清理最旧的群信息缓存条目，保留一半容量。"""
        sorted_items = sorted(
            self._group_cache.items(), key=lambda x: x[1].fetched_at
        )
        keep = _GROUP_CACHE_MAX // 2
        self._group_cache = dict(sorted_items[len(sorted_items) - keep:])

    def _evict_oldest_members(self) -> None:
        """清理最旧的成员信息缓存条目，保留一半容量。"""
        sorted_items = sorted(
            self._member_cache.items(), key=lambda x: x[1].fetched_at
        )
        keep = _MEMBER_CACHE_MAX // 2
        self._member_cache = dict(sorted_items[len(sorted_items) - keep:])
