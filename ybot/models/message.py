"""消息段数据模型。

定义 OneBot v11 的消息段（Message Segment）结构，
提供消息解析和可读文本转换功能。

消息段文本表示分为两种用途：
- segment_to_text / segments_to_text: 用于日志输出，简洁可读。
- segment_to_content / segments_to_content: 用于 LLM 上下文，携带结构化元信息。
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
    """将单个消息段转为可读文本（用于日志输出）。

    Args:
        seg: 消息段。

    Returns:
        可读文本表示。
    """
    match seg.type:
        case "text":
            return seg.data.get("text", "")
        case "image":
            sub_type = seg.data.get("sub_type")
            if sub_type == 1:
                return "[自定义表情]"
            return "[图片]"
        case "mface":
            summary = seg.data.get("summary", "")
            return f"[商城表情:{summary}]" if summary else "[商城表情]"
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
        case "file":
            name = seg.data.get("name", "")
            return f"[文件:{name}]" if name else "[文件]"
        case "share":
            title = seg.data.get("title", "链接")
            return f"[分享:{title}]"
        case "forward":
            return "[合并转发消息]"
        case "json":
            return "[JSON消息]"
        case "xml":
            return "[XML消息]"
        case "poke":
            return "[戳一戳]"
        case "dice":
            result = seg.data.get("result", "?")
            return f"[骰子:{result}]"
        case "rps":
            result = seg.data.get("result", "?")
            return f"[猜拳:{result}]"
        case "music":
            music_type = seg.data.get("type", "")
            return f"[音乐:{music_type}]" if music_type else "[音乐]"
        case "contact":
            return "[推荐联系人]"
        case "markdown":
            return "[Markdown消息]"
        case _:
            return f"[{seg.type}]"


def segments_to_text(segments: list[MessageSegment]) -> str:
    """将消息段列表转为可读文本（用于日志输出）。

    Args:
        segments: 消息段列表。

    Returns:
        拼接后的可读文本。
    """
    return "".join(segment_to_text(seg) for seg in segments)


# ---- LLM 上下文用：携带结构化元信息 ----


def segment_to_content(seg: MessageSegment) -> str:
    """将单个消息段转为 LLM 可感知的内容表示。

    与 segment_to_text 不同，此函数会携带消息段中的关键元信息
    （如文件名、file hash、图片摘要等），让 LLM 能感知更多上下文。

    格式约定：
    - text / at 段：与 segment_to_text 一致。
    - 其他段：``[类型 key:value ...]`` 格式，仅包含非空字段。

    Args:
        seg: 消息段。

    Returns:
        携带元信息的内容表示。
    """
    match seg.type:
        case "text":
            return seg.data.get("text", "")
        case "at":
            qq = seg.data.get("qq", "?")
            return f'<at qq="{qq}"/>'
        case "image":
            return _image_to_content(seg)
        case "mface":
            summary = seg.data.get("summary", "")
            return f'[商城表情:"{summary}"]' if summary else "[商城表情]"
        case "face":
            face_id = seg.data.get("id", "?")
            return f"[表情:id={face_id}]"
        case "record":
            return _file_based_to_content("语音", seg)
        case "video":
            return _file_based_to_content("视频", seg)
        case "file":
            return _file_based_to_content("文件", seg)
        case "reply":
            msg_id = seg.data.get("id", "?")
            return f"[回复:#{msg_id}]"
        case "forward":
            return "[合并转发消息]"
        case "dice":
            result = seg.data.get("result", "?")
            return f"[骰子:结果={result}]"
        case "rps":
            _RPS_MAP = {"1": "石头", "2": "剪刀", "3": "布"}
            result = str(seg.data.get("result", "?"))
            result_text = _RPS_MAP.get(result, result)
            return f"[猜拳:结果={result_text}]"
        case "share":
            title = seg.data.get("title", "")
            url = seg.data.get("url", "")
            parts = ["分享"]
            if title:
                parts.append(f'title:"{title}"')
            if url:
                parts.append(f"url:{url}")
            return "[" + " ".join(parts) + "]"
        case "json":
            return "[JSON卡片消息]"
        case "xml":
            return "[XML消息]"
        case "poke":
            return "[戳一戳]"
        case "music":
            music_type = seg.data.get("type", "")
            title = seg.data.get("title", "")
            parts = ["音乐卡片"]
            if music_type:
                parts.append(f"平台:{music_type}")
            if title:
                parts.append(f'title:"{title}"')
            return "[" + " ".join(parts) + "]"
        case "contact":
            contact_type = seg.data.get("type", "")
            contact_id = seg.data.get("id", "")
            return f"[推荐联系人 type:{contact_type} id:{contact_id}]"
        case "markdown":
            return "[Markdown消息]"
        case _:
            return f"[{seg.type}]"


def segments_to_content(segments: list[MessageSegment]) -> str:
    """将消息段列表转为 LLM 可感知的内容表示。

    Args:
        segments: 消息段列表。

    Returns:
        拼接后的内容文本（携带结构化元信息）。
    """
    return "".join(segment_to_content(seg) for seg in segments)


# ---- 内部辅助函数 ----


def _image_to_content(seg: MessageSegment) -> str:
    """将 image 消息段转为 LLM 内容表示。

    根据 sub_type 区分普通图片和自定义表情：
    - sub_type=0: 普通图片 (KNORMAL)
    - sub_type=1: 自定义表情 (KCUSTOM)
    - 其他值: 普通图片

    携带的元信息：name, summary, file（hash 标识）。
    """
    sub_type = seg.data.get("sub_type")
    label = "自定义表情" if sub_type == 1 else "图片"

    parts = [label]
    name = seg.data.get("name", "")
    summary = seg.data.get("summary", "")
    file_hash = seg.data.get("file", "")

    if name:
        parts.append(f'name:"{name}"')
    if summary and summary not in ("[图片]", ""):
        parts.append(f'summary:"{summary}"')
    if file_hash:
        parts.append(f"file:{file_hash}")

    return "[" + " ".join(parts) + "]"


def _file_based_to_content(label: str, seg: MessageSegment) -> str:
    """将基于 FileBase schema 的消息段（语音/视频/文件）转为 LLM 内容表示。

    FileBase 共有字段：path, thumb, name, file, url。
    仅携带 name 和 file（hash 标识）。
    """
    parts = [label]
    name = seg.data.get("name", "")
    file_hash = seg.data.get("file", "")

    if name:
        parts.append(f'name:"{name}"')
    if file_hash:
        parts.append(f"file:{file_hash}")

    return "[" + " ".join(parts) + "]"
