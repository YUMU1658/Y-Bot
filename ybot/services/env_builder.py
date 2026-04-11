"""ENV 头部构建与消息元信息格式化。

提供两个核心类：
- EnvBuilder: 构建 System Prompt 的 [ENV] 头部（群聊/私聊/临时会话）。
- MessageFormatter: 将每条用户消息包装为带发送者身份元信息的格式。
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from ybot.utils.logger import get_logger

if TYPE_CHECKING:
    from ybot.models.event import GroupMessageEvent, PrivateMessageEvent
    from ybot.services.bot_info import BotInfoService

logger = get_logger("ENV")

# 北京时间 UTC+8
_CST = timezone(timedelta(hours=8))

# 角色中文映射
_ROLE_MAP = {
    "owner": "群主",
    "admin": "管理员",
    "member": "成员",
}


def _build_identity_parts(
    *,
    level: str,
    title: str,
    role: str,
) -> str:
    """构建等级/头衔/身份部分，缺什么省什么（连同符号）。

    输出示例：
    - 完整: ``Lv.8(冒泡达人)「荣誉会员」| 管理员``
    - 无等级头衔: ``Lv.8「荣誉会员」| 管理员``
    - 无专属头衔: ``Lv.8(冒泡达人) | 管理员``
    - 仅角色: ``管理员``
    - 全空: ``""``

    Args:
        level: 等级字段原始值（可能是数字字符串或等级名称）。
        title: 专属头衔。
        role: 群角色 (owner/admin/member)。

    Returns:
        格式化后的身份字符串。
    """
    parts: list[str] = []

    # 等级部分
    if level:
        if level.isdigit():
            # 纯数字 → Lv.{数字}
            parts.append(f"Lv.{level}")
        else:
            # 包含非数字字符 → 尝试提取数字部分
            digits = "".join(c for c in level if c.isdigit())
            if digits:
                parts.append(f"Lv.{digits}({level})")
            else:
                # 纯文字等级名称
                parts.append(f"({level})")

    # 专属头衔
    if title:
        parts.append(f"「{title}」")

    # 角色
    role_text = _ROLE_MAP.get(role, "")
    if role_text:
        parts.append(role_text)

    if not parts:
        return ""

    # 用 " | " 分隔等级/头衔组合 与 角色
    # 等级和头衔紧挨，角色用 | 分隔
    identity_prefix = ""
    if level or title:
        level_title_parts: list[str] = []
        if level:
            if level.isdigit():
                level_title_parts.append(f"Lv.{level}")
            else:
                digits = "".join(c for c in level if c.isdigit())
                if digits:
                    level_title_parts.append(f"Lv.{digits}({level})")
                else:
                    level_title_parts.append(f"({level})")
        if title:
            level_title_parts.append(f"「{title}」")
        identity_prefix = "".join(level_title_parts)

    if identity_prefix and role_text:
        return f"{identity_prefix} | {role_text}"
    elif identity_prefix:
        return identity_prefix
    else:
        return role_text


class EnvBuilder:
    """构建 System Prompt 的 [ENV] 头部。

    根据会话类型（群聊/私聊/临时会话）和实时数据，
    生成包含当前环境信息的文本块。
    """

    def __init__(self, bot_info: BotInfoService) -> None:
        self._bot_info = bot_info

    async def build_group_env(self, group_id: int) -> str:
        """构建群聊 ENV 头部。

        格式::

            [ENV]
            Group: {群名}(ID:{group_id}) | {群人数}人
            Time: {YYYY-MM-DD HH:MM} (UTC+8)
            Self: @{Bot昵称} → {Bot群昵称} | {等级/头衔/身份}

        Args:
            group_id: 群号。

        Returns:
            ENV 头部文本。
        """
        # 获取群信息
        group_info = await self._bot_info.get_group_info(group_id)

        # 获取 Bot 登录信息
        login_info = await self._bot_info.get_login_info()
        bot_nickname = login_info.nickname or str(login_info.user_id)

        # 获取 Bot 在群内的成员信息
        bot_member = await self._bot_info.get_member_info(group_id, login_info.user_id)

        # 构建 Group 行
        group_name = group_info.group_name
        member_count = group_info.member_count
        if group_name and member_count:
            group_line = f"Group: {group_name}(ID:{group_id}) | {member_count}人"
        elif group_name:
            group_line = f"Group: {group_name}(ID:{group_id})"
        else:
            group_line = f"Group: (ID:{group_id})"

        # 构建 Time 行
        now = datetime.now(_CST)
        time_line = f"Time: {now.strftime('%Y-%m-%d %H:%M')} (UTC+8)"

        # 构建 Self 行
        self_line = self._build_self_line(
            bot_nickname=bot_nickname,
            bot_card=bot_member.card,
            level=bot_member.level,
            title=bot_member.title,
            role=bot_member.role,
        )

        return f"[ENV]\n{group_line}\n{time_line}\n{self_line}"

    async def build_private_env(self, user_id: int) -> str:
        """构建私聊 ENV 头部。

        格式::

            [ENV]
            Friend(ID:friend_{user_id})
            Time: {YYYY-MM-DD HH:MM} (UTC+8)
            Self: @{Bot昵称}

        Args:
            user_id: 对方用户 QQ 号。

        Returns:
            ENV 头部文本。
        """
        login_info = await self._bot_info.get_login_info()
        bot_nickname = login_info.nickname or str(login_info.user_id)

        now = datetime.now(_CST)
        time_line = f"Time: {now.strftime('%Y-%m-%d %H:%M')} (UTC+8)"

        return f"[ENV]\nFriend(ID:friend_{user_id})\n{time_line}\nSelf: @{bot_nickname}"

    async def build_temp_env(self, user_id: int, source_group_id: int) -> str:
        """构建临时会话 ENV 头部。

        格式::

            [ENV]
            Temp(ID:temp_{source_group_id}_{user_id}) ← {来源群名}(ID:{source_group_id})
            Time: {YYYY-MM-DD HH:MM} (UTC+8)
            Self: @{Bot昵称}

        Args:
            user_id: 对方用户 QQ 号。
            source_group_id: 来源群号。

        Returns:
            ENV 头部文本。
        """
        login_info = await self._bot_info.get_login_info()
        bot_nickname = login_info.nickname or str(login_info.user_id)

        # 获取来源群名
        group_info = await self._bot_info.get_group_info(source_group_id)
        group_name = group_info.group_name

        now = datetime.now(_CST)
        time_line = f"Time: {now.strftime('%Y-%m-%d %H:%M')} (UTC+8)"

        if group_name:
            temp_line = (
                f"Temp(ID:temp_{source_group_id}_{user_id})"
                f" \u2190 {group_name}(ID:{source_group_id})"
            )
        else:
            temp_line = (
                f"Temp(ID:temp_{source_group_id}_{user_id})"
                f" \u2190 (ID:{source_group_id})"
            )

        return f"[ENV]\n{temp_line}\n{time_line}\nSelf: @{bot_nickname}"

    @staticmethod
    def _build_self_line(
        *,
        bot_nickname: str,
        bot_card: str,
        level: str,
        title: str,
        role: str,
    ) -> str:
        """构建 Self 行。

        完整格式: ``Self: @{Bot昵称} → {Bot群昵称} | {等级/头衔/身份}``
        缺省规则：缺什么省什么，连同分隔符。
        """
        parts = [f"Self: @{bot_nickname}"]

        # 群昵称（名片）
        if bot_card and bot_card != bot_nickname:
            parts.append(f" \u2192 {bot_card}")

        # 身份部分
        identity = _build_identity_parts(level=level, title=title, role=role)
        if identity:
            parts.append(f" | {identity}")

        return "".join(parts)


class MessageFormatter:
    """格式化消息元信息。

    将每条用户消息包装为带发送者身份信息的格式，
    存入数据库供 LLM 感知历史消息的发送者身份。
    """

    def __init__(self, bot_info: BotInfoService) -> None:
        self._bot_info = bot_info

    async def format_group_message(self, event: GroupMessageEvent, text: str) -> str:
        """格式化群聊消息为带元信息的文本。

        输出格式::

            [#{msg_id} {QQ昵称}({QQ号}) → {群昵称} | {等级/头衔/身份} ★友]
            {消息内容}

        缺省规则：缺什么删什么，连同符号。

        Args:
            event: 群聊消息事件。
            text: 提取的纯文本消息内容。

        Returns:
            格式化后的消息文本。
        """
        msg_id = event.message_id
        user_id = event.user_id

        # 从事件获取基础信息（作为降级数据）
        event_nickname = event.sender.nickname or str(user_id)
        event_card = event.sender.card or ""
        event_role = event.sender.role or ""

        # 尝试从 API 获取完整成员信息
        member = await self._bot_info.get_member_info(event.group_id, user_id)

        # 优先使用 API 数据，降级到事件数据
        nickname = member.nickname or event_nickname
        card = member.card or event_card
        role = member.role or event_role
        level = member.level
        title = member.title

        # 构建元信息头
        header_parts = [f"#{msg_id} {nickname}({user_id})"]

        # 群昵称
        if card and card != nickname:
            header_parts.append(f" \u2192 {card}")

        # 身份部分
        identity = _build_identity_parts(level=level, title=title, role=role)
        if identity:
            header_parts.append(f" | {identity}")

        # ★友 标记
        is_friend = await self._bot_info.is_friend(user_id)
        if is_friend:
            header_parts.append(" \u2605友")

        header = "[" + "".join(header_parts) + "]"
        return f"{header}\n{text}"

    @staticmethod
    def format_private_message(event: PrivateMessageEvent, text: str) -> str:
        """格式化私聊消息为带元信息的文本。

        输出格式::

            [#{msg_id}]
            {消息内容}

        Args:
            event: 私聊消息事件。
            text: 提取的纯文本消息内容。

        Returns:
            格式化后的消息文本。
        """
        return f"[#{event.message_id}]\n{text}"
