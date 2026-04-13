"""消息段数据模型。

定义 OneBot v11 的消息段（Message Segment）结构，
提供消息解析和可读文本转换功能。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MessageSegment:
    """OneBot v11 消息段。

    Attributes:
        type: 消息段类型（text, image, face, at, reply 等）。
        data: 消息段数据。
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """返回消息段的可读文本表示。"""
        return segment_to_text(self)


def parse_message(message: list[dict[str, Any]]) -> list[MessageSegment]:
    """将 OneBot 的 message 数组解析为 MessageSegment 列表。

    Args:
        message: OneBot 事件中的 message 字段（数组格式）。

    Returns:
        MessageSegment 列表。
    """
    segments: list[MessageSegment] = []
    for seg in message:
        segments.append(
            MessageSegment(
                type=seg.get("type", "unknown"),
                data=seg.get("data", {}),
            )
        )
    return segments


def segment_to_text(seg: MessageSegment) -> str:
    """将单个消息段转为可读文本。

    Args:
        seg: 消息段。

    Returns:
        可读文本表示。
    """
    match seg.type:
        case "text":
            return seg.data.get("text", "")
        case "image":
            return "[图片]"
        case "face":
            face_id = seg.data.get("id", "?")
            return f"[表情:{face_id}]"
        case "at":
            qq = seg.data.get("qq", "?")
            return f'<at qq="{qq}"/>'
        case "reply":
            msg_id = seg.data.get("id", "?")
            return f"[回复:{msg_id}]"
        case "record":
            return "[语音]"
        case "video":
            return "[视频]"
        case "share":
            title = seg.data.get("title", "链接")
            return f"[分享:{title}]"
        case "forward":
            return "[合并转发]"
        case "json":
            return "[JSON消息]"
        case "xml":
            return "[XML消息]"
        case "poke":
            return "[戳一戳]"
        case _:
            return f"[{seg.type}]"


def segments_to_text(segments: list[MessageSegment]) -> str:
    """将消息段列表转为可读文本。

    Args:
        segments: 消息段列表。

    Returns:
        拼接后的可读文本。
    """
    return "".join(segment_to_text(seg) for seg in segments)
