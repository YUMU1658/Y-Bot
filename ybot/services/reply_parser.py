"""AI 回复解析模块。

解析 AI 回复中的 <send_msg> 标签，提取需要发送的消息列表。
支持可选的 reply_id 属性用于引用/回复消息。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 匹配 <send_msg> 或 <send_msg reply_id="...">，允许内容换行
_SEND_MSG_PATTERN = re.compile(
    r'<send_msg(?:\s+reply_id="(\d+)")?\s*>(.*?)</send_msg>', re.DOTALL
)


@dataclass
class ParsedMessage:
    """解析后的单条消息。"""

    content: str
    reply_id: int | None = None


def parse_reply(raw_reply: str) -> list[ParsedMessage]:
    """解析 AI 回复，提取所有 <send_msg> 标签中的消息。

    Args:
        raw_reply: AI 原始回复文本。

    Returns:
        解析后的消息列表（按出现顺序）。
        如果没有 <send_msg> 标签，返回空列表。
    """
    results: list[ParsedMessage] = []
    for match in _SEND_MSG_PATTERN.finditer(raw_reply):
        reply_id_str = match.group(1)  # 可能为 None
        content = match.group(2).strip()
        reply_id = int(reply_id_str) if reply_id_str else None

        # 有 reply_id 时允许空内容（QQ 支持纯引用消息）
        # 无 reply_id 时过滤空内容（保持原有行为）
        if not content and reply_id is None:
            continue

        results.append(ParsedMessage(content=content, reply_id=reply_id))
    return results
