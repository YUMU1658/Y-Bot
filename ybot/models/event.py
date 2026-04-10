"""事件数据模型。

基于 OneBot v11 事件结构定义数据类，
提供事件解析工厂函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ybot.models.message import MessageSegment, parse_message


@dataclass
class Sender:
    """消息发送者信息。"""

    user_id: int = 0
    nickname: str = ""
    card: str = ""  # 群名片
    sex: str = "unknown"
    age: int = 0
    role: str = ""  # 群角色: owner / admin / member

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sender:
        return cls(
            user_id=data.get("user_id", 0),
            nickname=data.get("nickname", ""),
            card=data.get("card", ""),
            sex=data.get("sex", "unknown"),
            age=data.get("age", 0),
            role=data.get("role", ""),
        )


@dataclass
class Event:
    """OneBot v11 事件基类。"""

    time: int = 0
    self_id: int = 0
    post_type: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageEvent(Event):
    """消息事件基类。"""

    message_type: str = ""  # private / group
    sub_type: str = ""  # friend / group / normal 等
    message_id: int = 0
    user_id: int = 0
    message: list[MessageSegment] = field(default_factory=list)
    raw_message: str = ""
    font: int = 0
    sender: Sender = field(default_factory=Sender)


@dataclass
class PrivateMessageEvent(MessageEvent):
    """私聊消息事件。"""

    def __post_init__(self) -> None:
        self.message_type = "private"


@dataclass
class GroupMessageEvent(MessageEvent):
    """群聊消息事件。"""

    group_id: int = 0

    def __post_init__(self) -> None:
        self.message_type = "group"


@dataclass
class NoticeEvent(Event):
    """通知事件。"""

    notice_type: str = ""


@dataclass
class RequestEvent(Event):
    """请求事件。"""

    request_type: str = ""


@dataclass
class MetaEvent(Event):
    """元事件（心跳、生命周期等）。"""

    meta_event_type: str = ""


def parse_event(data: dict[str, Any]) -> Event:
    """将原始 JSON 数据解析为对应的事件对象。

    Args:
        data: OneBot 上报的原始 JSON 数据（已解析为 dict）。

    Returns:
        对应类型的 Event 实例。
    """
    post_type = data.get("post_type", "")

    # 公共字段
    base_kwargs: dict[str, Any] = {
        "time": data.get("time", 0),
        "self_id": data.get("self_id", 0),
        "post_type": post_type,
        "raw_data": data,
    }

    match post_type:
        case "message" | "message_sent":
            return _parse_message_event(data, base_kwargs)
        case "notice":
            return NoticeEvent(
                **base_kwargs,
                notice_type=data.get("notice_type", ""),
            )
        case "request":
            return RequestEvent(
                **base_kwargs,
                request_type=data.get("request_type", ""),
            )
        case "meta_event":
            return MetaEvent(
                **base_kwargs,
                meta_event_type=data.get("meta_event_type", ""),
            )
        case _:
            return Event(**base_kwargs)


def _parse_message_event(
    data: dict[str, Any],
    base_kwargs: dict[str, Any],
) -> MessageEvent:
    """解析消息事件。"""
    message_type = data.get("message_type", "")
    raw_message_data = data.get("message", [])

    # 消息段解析
    if isinstance(raw_message_data, list):
        segments = parse_message(raw_message_data)
    else:
        segments = []

    sender = Sender.from_dict(data.get("sender", {}))

    common_kwargs = {
        **base_kwargs,
        "message_type": message_type,
        "sub_type": data.get("sub_type", ""),
        "message_id": data.get("message_id", 0),
        "user_id": data.get("user_id", 0),
        "message": segments,
        "raw_message": data.get("raw_message", ""),
        "font": data.get("font", 0),
        "sender": sender,
    }

    match message_type:
        case "group":
            return GroupMessageEvent(
                **common_kwargs,
                group_id=data.get("group_id", 0),
            )
        case "private":
            return PrivateMessageEvent(**common_kwargs)
        case _:
            return MessageEvent(**common_kwargs)
