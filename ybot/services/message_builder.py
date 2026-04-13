"""消息构建模块。

将包含 <at qq="..."/> 标签的文本转换为 OneBot v11 消息段数组。
"""

import re
from typing import Any

# 匹配 <at qq="..."/> 标签
_AT_PATTERN = re.compile(r'<at\s+qq="([^"]+)"\s*/>')


def text_to_segments(text: str) -> list[dict[str, Any]]:
    """将包含 <at> 标签的文本转换为 OneBot 消息段数组。

    解析文本中的 <at qq="..."/> 标签，将其转换为 at 类型消息段，
    其余文本转换为 text 类型消息段。

    Args:
        text: 可能包含 <at qq="..."/> 标签的文本。

    Returns:
        OneBot v11 消息段数组（list of dict）。
    """
    segments: list[dict[str, Any]] = []
    last_end = 0

    for match in _AT_PATTERN.finditer(text):
        # 添加 at 标签前的文本段
        before = text[last_end : match.start()]
        if before:
            segments.append({"type": "text", "data": {"text": before}})

        # 添加 at 段
        qq = match.group(1)
        segments.append({"type": "at", "data": {"qq": qq}})

        last_end = match.end()

    # 添加最后一段文本
    remaining = text[last_end:]
    if remaining:
        segments.append({"type": "text", "data": {"text": remaining}})

    return segments
