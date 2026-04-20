"""群聊消息日志缓冲区。

收集群内所有消息（包括 bot 自身发出的），
在 bot 被触发时提供近期聊天记录作为上下文。
纯内存实现，使用 deque 自动淘汰旧记录。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class ChatLogEntry:
    """单条聊天记录。

    Attributes:
        text: 消息内容表示。包含纯文本以及富媒体占位标记
            （如 ``[图片 file:xxx]``、``[语音]``、``[商城表情:"可爱"]`` 等），
            由 ``segments_to_content()`` 生成。
    """

    message_id: int  # OneBot message_id（去重依据）
    group_id: int
    user_id: int
    nickname: str
    card: str  # 群名片
    role: str  # 群角色 (owner/admin/member)
    level: str  # 群等级
    title: str  # 专属头衔
    is_friend: bool
    timestamp: float  # Unix 时间戳（来自 event.time）
    text: str  # 消息内容表示（含富媒体占位标记）
    is_bot: bool  # 是否为 bot 自身发送的消息
    recalled: bool = False  # 是否已被撤回


class GroupChatLog:
    """群聊消息日志管理器。

    每个群维护一个有序的消息列表（按时间排序），
    内存中保留最近 ``buffer_size`` 条记录。
    """

    def __init__(self, buffer_size: int = 100) -> None:
        self._buffer_size = buffer_size
        self._logs: dict[int, deque[ChatLogEntry]] = {}

    def add(self, entry: ChatLogEntry) -> None:
        """添加一条聊天记录。"""
        gid = entry.group_id
        if gid not in self._logs:
            self._logs[gid] = deque(maxlen=self._buffer_size)
        self._logs[gid].append(entry)

    def get_recent(self, group_id: int, limit: int = 20) -> list[ChatLogEntry]:
        """获取最近 N 条聊天记录。

        Args:
            group_id: 群号。
            limit: 最大返回条数。

        Returns:
            按时间升序排列的聊天记录列表。
        """
        if group_id not in self._logs:
            return []
        entries = list(self._logs[group_id])
        return entries[-limit:] if len(entries) > limit else entries

    def get_between(
        self,
        group_id: int,
        after_msg_id: int | None,
        before_msg_id: int,
        limit: int = 20,
    ) -> list[ChatLogEntry]:
        """获取 after_msg_id 之后、before_msg_id 之前的记录（不含两端）。

        如果 after_msg_id 为 None，则从最早可用记录开始。
        返回最多 limit 条，优先取最近的。

        Args:
            group_id: 群号。
            after_msg_id: 起始 message_id（不含），None 表示从头开始。
            before_msg_id: 结束 message_id（不含），通常为当前触发消息的 ID。
            limit: 最大返回条数。

        Returns:
            按时间升序排列的聊天记录列表。
        """
        if group_id not in self._logs:
            return []

        entries = list(self._logs[group_id])

        # 找到 after_msg_id 的位置
        start_idx = 0
        if after_msg_id is not None:
            for i, e in enumerate(entries):
                if e.message_id == after_msg_id:
                    start_idx = i + 1
                    break

        # 找到 before_msg_id 的位置
        end_idx = len(entries)
        for i, e in enumerate(entries):
            if e.message_id == before_msg_id:
                end_idx = i
                break

        result = entries[start_idx:end_idx]
        return result[-limit:] if len(result) > limit else result

    def mark_recalled(self, message_id: int) -> None:
        """将指定消息标记为已撤回。

        遍历所有群的缓冲区，找到对应 message_id 的条目并标记。
        message_id 全局唯一，找到后立即返回。

        Args:
            message_id: 被撤回消息的 message_id。
        """
        for entries in self._logs.values():
            for entry in entries:
                if entry.message_id == message_id:
                    entry.recalled = True
                    return

    def is_recalled(self, message_id: int) -> bool:
        """查询指定消息是否已被标记为撤回。

        Args:
            message_id: 消息的 message_id。

        Returns:
            如果消息在缓冲区中且已标记为撤回则返回 True，否则返回 False。
        """
        for entries in self._logs.values():
            for entry in entries:
                if entry.message_id == message_id:
                    return entry.recalled
        return False
