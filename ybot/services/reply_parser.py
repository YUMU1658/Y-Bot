"""AI 回复解析模块。

解析 AI 回复中的 <send_msg> 标签，提取需要发送的消息列表。
"""

from __future__ import annotations

import re

# 匹配 <send_msg>...</send_msg> 标签，允许内容换行（re.DOTALL）
_SEND_MSG_PATTERN = re.compile(r"<send_msg>(.*?)</send_msg>", re.DOTALL)


def parse_reply(raw_reply: str) -> list[str]:
    """解析 AI 回复，提取所有 <send_msg> 标签中的消息。

    Args:
        raw_reply: AI 原始回复文本。

    Returns:
        提取的消息列表（按出现顺序）。
        如果没有 <send_msg> 标签，返回空列表。
    """
    matches = _SEND_MSG_PATTERN.findall(raw_reply)
    # 去除每条消息首尾空白，过滤空消息
    return [msg.strip() for msg in matches if msg.strip()]
