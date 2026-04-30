"""对话存储模块。

使用 SQLite 存储多轮对话历史，按 session_key 隔离不同会话。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import aiosqlite

from ybot.utils.logger import get_logger

logger = get_logger("存储")

# 建表 SQL
_CREATE_TABLES_SQL = """\
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT    UNIQUE NOT NULL,
    created_at  REAL   NOT NULL,
    updated_at  REAL   NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL,
    role             TEXT    NOT NULL,
    content          TEXT    NOT NULL,
    content_type     TEXT    NOT NULL DEFAULT 'text',
    created_at       REAL   NOT NULL,
    last_ref_msg_id  INTEGER DEFAULT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session_time
    ON messages (session_id, created_at);
"""

# 迁移 SQL：为已有数据库添加 last_ref_msg_id 列
_MIGRATION_ADD_LAST_REF_MSG_ID = (
    "ALTER TABLE messages ADD COLUMN last_ref_msg_id INTEGER DEFAULT NULL"
)

# 迁移 SQL：为已有数据库添加 content_type 列
_MIGRATION_ADD_CONTENT_TYPE = (
    "ALTER TABLE messages ADD COLUMN content_type TEXT NOT NULL DEFAULT 'text'"
)

# 迁移 SQL：为已有数据库添加 last_invoked_at 列（跨会话记忆）
_MIGRATION_ADD_LAST_INVOKED_AT = (
    "ALTER TABLE sessions ADD COLUMN last_invoked_at REAL DEFAULT NULL"
)

# 迁移 SQL：为已有数据库添加 display_name 列（跨会话记忆）
_MIGRATION_ADD_DISPLAY_NAME = (
    "ALTER TABLE sessions ADD COLUMN display_name TEXT DEFAULT NULL"
)


class ConversationStore:
    """对话存储层，封装所有 SQLite 操作。

    Attributes:
        _db_path: 数据库文件路径。
        _db: aiosqlite 数据库连接。
    """

    def __init__(self, db_path: str = "data/conversations.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """创建数据库连接并建表（IF NOT EXISTS）。

        自动创建 data/ 目录（如果不存在）。
        对已有数据库执行必要的 schema 迁移。
        """
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(_CREATE_TABLES_SQL)
        await self._db.commit()

        # 迁移：为已有数据库添加 last_ref_msg_id 列（如果不存在）
        try:
            await self._db.execute(_MIGRATION_ADD_LAST_REF_MSG_ID)
            await self._db.commit()
        except Exception:
            # 列已存在时 SQLite 会抛出 OperationalError，忽略即可
            pass

        # 迁移：为已有数据库添加 content_type 列（如果不存在）
        try:
            await self._db.execute(_MIGRATION_ADD_CONTENT_TYPE)
            await self._db.commit()
        except Exception:
            pass

        # 迁移：为已有数据库添加 last_invoked_at 列（跨会话记忆）
        try:
            await self._db.execute(_MIGRATION_ADD_LAST_INVOKED_AT)
            await self._db.commit()
        except Exception:
            pass

        # 迁移：为已有数据库添加 display_name 列（跨会话记忆）
        try:
            await self._db.execute(_MIGRATION_ADD_DISPLAY_NAME)
            await self._db.commit()
        except Exception:
            pass

        logger.info(f"对话存储已初始化: {self._db_path}")

    async def close(self) -> None:
        """关闭数据库连接。"""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("对话存储已关闭")

    async def add_message(
        self,
        session_key: str,
        role: str,
        content: str,
        last_ref_msg_id: int | None = None,
        content_type: str = "text",
    ) -> None:
        """向指定会话添加一条消息。如果会话不存在则自动创建。

        Args:
            session_key: 会话标识（如 ``friend_12345``、``group_67890`` 或 ``temp_11111_22222``）。
            role: 消息角色（``user`` / ``assistant`` / ``tool``）。
            content: 消息文本内容。当 content_type 为 ``multimodal``、``tool_calls``
                或 ``tool_result`` 时，应为 JSON 字符串。
            last_ref_msg_id: 本轮参考聊天记录中最新一条的 message_id（仅 role=user 时有意义）。
            content_type: 内容类型（``text`` / ``multimodal`` / ``tool_calls`` / ``tool_result``）。
        """
        assert self._db is not None, "ConversationStore 未初始化"

        now = time.time()

        # 确保 session 存在（INSERT OR IGNORE + UPDATE updated_at）
        await self._db.execute(
            "INSERT OR IGNORE INTO sessions (session_key, created_at, updated_at) "
            "VALUES (?, ?, ?)",
            (session_key, now, now),
        )
        await self._db.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_key = ?",
            (now, session_key),
        )

        # 获取 session_id
        cursor = await self._db.execute(
            "SELECT id FROM sessions WHERE session_key = ?",
            (session_key,),
        )
        row = await cursor.fetchone()
        assert row is not None
        session_id: int = row[0]

        # 插入消息
        await self._db.execute(
            "INSERT INTO messages (session_id, role, content, content_type, created_at, last_ref_msg_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, content_type, now, last_ref_msg_id),
        )
        await self._db.commit()

    async def get_history(
        self, session_key: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """获取指定会话的最近 N 条消息。

        对于 ``content_type="multimodal"`` 的消息，会将 content 从 JSON 字符串
        解析为列表，并将其中的 ``image_url`` 元素替换为文本占位符 ``[图片(历史)]``，
        避免回放过期 URL 和浪费 token。

        对于 ``content_type="tool_calls"`` 的消息，还原为包含 ``tool_calls`` 键的
        assistant 消息。对于 ``content_type="tool_result"`` 的消息，还原为 ``role="tool"``
        的消息。返回前会调用 ``_trim_orphan_tool_messages()`` 修剪截断导致的不完整序列。

        Args:
            session_key: 会话标识。
            limit: 最大返回消息数。

        Returns:
            按时间升序排列的消息列表，每条为
            ``{"role": "user"|"assistant"|"tool", "content": "..." | [...]}``。
        """
        assert self._db is not None, "ConversationStore 未初始化"

        # 子查询取最近 N 条（DESC），外层再按时间正序排列
        cursor = await self._db.execute(
            "SELECT role, content, content_type FROM ("
            "  SELECT m.role, m.content, m.content_type, m.created_at "
            "  FROM messages m "
            "  JOIN sessions s ON m.session_id = s.id "
            "  WHERE s.session_key = ? "
            "  ORDER BY m.created_at DESC "
            "  LIMIT ?"
            ") ORDER BY created_at ASC",
            (session_key, limit),
        )
        rows = await cursor.fetchall()

        result: list[dict[str, Any]] = []
        for r in rows:
            role, content, content_type = r[0], r[1], r[2]
            if content_type == "tool_calls":
                # 还原为 assistant + tool_calls 格式
                try:
                    data = json.loads(content)
                    msg: dict[str, Any] = {"role": "assistant", "content": data.get("content")}
                    msg["tool_calls"] = data["tool_calls"]
                    result.append(msg)
                except (json.JSONDecodeError, TypeError, KeyError):
                    # 解析失败时降级为纯文本
                    result.append({"role": role, "content": content})
            elif content_type == "tool_result":
                # 还原为 tool role 消息
                try:
                    data = json.loads(content)
                    result.append({
                        "role": "tool",
                        "tool_call_id": data["tool_call_id"],
                        "content": data["content"],
                    })
                except (json.JSONDecodeError, TypeError, KeyError):
                    result.append({"role": role, "content": content})
            elif content_type == "multimodal":
                try:
                    content_list = json.loads(content)
                    # 将 image_url 元素替换为文本占位符
                    sanitized: list[dict[str, Any]] = []
                    for item in content_list:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            sanitized.append({"type": "text", "text": "[图片(历史)]"})
                        else:
                            sanitized.append(item)
                    result.append({"role": role, "content": sanitized})
                except (json.JSONDecodeError, TypeError):
                    # JSON 解析失败时降级为纯文本
                    result.append({"role": role, "content": content})
            else:
                result.append({"role": role, "content": content})

        # 修剪截断导致的不完整工具调用序列
        result = self._trim_orphan_tool_messages(result)
        return result

    @staticmethod
    def _trim_orphan_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """修剪历史开头不完整的工具调用序列。

        当 max_history 截断导致 tool_calls 和 tool_result 不成对时，
        移除开头的孤立消息，确保发送给 API 的消息序列合法。

        Args:
            messages: 按时间升序排列的消息列表。

        Returns:
            修剪后的消息列表。
        """
        start = 0
        while start < len(messages):
            msg = messages[start]
            role = msg.get("role", "")
            if role == "tool":
                # 孤立的 tool result（缺少前面的 assistant tool_calls），跳过
                start += 1
                continue
            if role == "assistant" and "tool_calls" in msg:
                # assistant tool_calls 消息，检查后续是否有完整的 tool results
                expected_ids = {tc["id"] for tc in msg["tool_calls"]}
                found_ids: set[str] = set()
                j = start + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    tid = messages[j].get("tool_call_id", "")
                    if tid:
                        found_ids.add(tid)
                    j += 1
                if expected_ids == found_ids:
                    break  # 完整的工具调用序列，保留
                else:
                    start = j  # 不完整，跳过整个序列
                    continue
            break  # 普通 user/assistant 消息，保留

        return messages[start:] if start > 0 else messages

    async def get_last_ref_msg_id(self, session_key: str) -> int | None:
        """获取指定会话最近一条 user 消息的 last_ref_msg_id。

        用于跨轮去重：下一轮构建参考记录时，从该 ID 之后开始取。

        Args:
            session_key: 会话标识。

        Returns:
            最近一条 user 消息的 last_ref_msg_id，如果不存在则返回 None。
        """
        assert self._db is not None, "ConversationStore 未初始化"

        cursor = await self._db.execute(
            "SELECT m.last_ref_msg_id "
            "FROM messages m "
            "JOIN sessions s ON m.session_id = s.id "
            "WHERE s.session_key = ? AND m.role = 'user' "
            "ORDER BY m.created_at DESC "
            "LIMIT 1",
            (session_key,),
        )
        row = await cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    async def update_session_meta(
        self, session_key: str, display_name: str | None = None
    ) -> None:
        """更新会话的唤醒时间和显示名称。

        Args:
            session_key: 会话标识。
            display_name: 会话显示名称（如群名、好友昵称）。
        """
        assert self._db is not None, "ConversationStore 未初始化"

        now = time.time()
        if display_name is not None:
            await self._db.execute(
                "UPDATE sessions SET last_invoked_at = ?, display_name = ? "
                "WHERE session_key = ?",
                (now, display_name, session_key),
            )
        else:
            await self._db.execute(
                "UPDATE sessions SET last_invoked_at = ? WHERE session_key = ?",
                (now, session_key),
            )
        await self._db.commit()

    async def get_recent_other_sessions(
        self,
        current_session_key: str,
        max_sessions: int = 5,
        decay_limits: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """获取最近被唤醒的其他会话的压缩记录。

        按唤醒时间降序排列，每个会话按衰减限制获取最近 N 条消息。

        Args:
            current_session_key: 当前会话标识（排除自身）。
            max_sessions: 最大返回会话数。
            decay_limits: 每个旧会话保留的消息条数（按远近递减）。

        Returns:
            按唤醒时间降序排列的会话列表，每个包含：
            - session_key: 会话标识
            - display_name: 会话显示名称
            - last_invoked_at: 最后唤醒时间戳
            - messages: 该会话的最近消息列表
        """
        assert self._db is not None, "ConversationStore 未初始化"

        if decay_limits is None:
            decay_limits = [20, 15, 10, 5, 3]

        # 查询最近唤醒的其他会话
        cursor = await self._db.execute(
            "SELECT session_key, display_name, last_invoked_at FROM sessions "
            "WHERE session_key != ? AND last_invoked_at IS NOT NULL "
            "ORDER BY last_invoked_at DESC LIMIT ?",
            (current_session_key, max_sessions),
        )
        sessions = await cursor.fetchall()

        result: list[dict[str, Any]] = []
        for i, (sk, name, invoked_at) in enumerate(sessions):
            limit = decay_limits[i] if i < len(decay_limits) else decay_limits[-1]
            messages = await self.get_history(sk, limit=limit)
            if messages:  # 跳过空会话
                result.append(
                    {
                        "session_key": sk,
                        "display_name": name or sk,  # 降级到 session_key
                        "last_invoked_at": invoked_at,
                        "messages": messages,
                    }
                )
        return result

    async def clear_session(self, session_key: str) -> None:
        """清空指定会话的所有消息（保留 session 记录）。

        Args:
            session_key: 会话标识。
        """
        assert self._db is not None, "ConversationStore 未初始化"

        cursor = await self._db.execute(
            "SELECT id FROM sessions WHERE session_key = ?",
            (session_key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return

        session_id: int = row[0]
        await self._db.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,),
        )
        await self._db.commit()

    async def update_last_assistant_message(
        self, session_key: str, new_content: str
    ) -> None:
        """更新指定会话最后一条 assistant 消息的内容。

        用于在 poke 标签执行后，将原始 <poke> 标签替换为执行结果文案。

        Args:
            session_key: 会话标识。
            new_content: 替换后的消息内容。
        """
        assert self._db is not None, "ConversationStore 未初始化"

        cursor = await self._db.execute(
            "SELECT m.id FROM messages m "
            "JOIN sessions s ON m.session_id = s.id "
            "WHERE s.session_key = ? AND m.role = 'assistant' "
            "ORDER BY m.created_at DESC LIMIT 1",
            (session_key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return

        msg_id: int = row[0]
        await self._db.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            (new_content, msg_id),
        )
        await self._db.commit()
