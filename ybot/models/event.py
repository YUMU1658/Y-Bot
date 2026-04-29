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
class PokeNoticeEvent(NoticeEvent):
    """戳一戳通知事件（群聊/私聊通用）。"""

    group_id: int = 0      # 群号（私聊时为 0）
    user_id: int = 0       # 戳者
    target_id: int = 0     # 被戳者
    poke_text: str = ""    # 互动文案（如"戳了戳"、"抱了抱XX并揉了揉"）


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
            notice_type = data.get("notice_type", "")
            sub_type = data.get("sub_type", "")
            if notice_type == "notify" and sub_type == "poke":
                return _parse_poke_event(data, base_kwargs)
            return NoticeEvent(
                **base_kwargs,
                notice_type=notice_type,
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


def _parse_poke_text(data: dict[str, Any]) -> str:
    """从戳一戳事件数据中解析互动文案。

    解析优先级：
    1. ``raw_data`` 顶层的 ``action`` 字段（部分 OneBot 实现如 go-cqhttp 扩展）
    2. ``raw_info`` 列表中提取（NapCat 扩展）
    3. 默认 "戳了戳"

    ``suffix`` 字段（如果存在）会追加到动作词后面。

    Args:
        data: 原始事件 JSON 数据。

    Returns:
        互动文案字符串，如 "戳了戳"、"抱了抱 并揉了揉"。
    """
    action = ""
    suffix = ""

    # 优先级 1：顶层 action 字段
    if data.get("action"):
        action = str(data["action"])
        suffix = str(data.get("suffix", ""))
    else:
        # 优先级 2：从 raw_info 提取
        raw_info = data.get("raw_info")
        if isinstance(raw_info, list):
            for item in raw_info:
                if not isinstance(item, dict):
                    continue
                # NapCat 的 raw_info 中可能包含 txt 字段作为动作词
                txt = item.get("txt", "")
                if txt and not action:
                    action = txt
        elif isinstance(raw_info, dict):
            action = raw_info.get("action", "") or raw_info.get("txt", "")
            suffix = raw_info.get("suffix", "")

    # 优先级 3：默认文案
    if not action:
        action = "戳了戳"

    if suffix:
        return f"{action} {suffix}".strip()
    return action


def _parse_poke_event(
    data: dict[str, Any],
    base_kwargs: dict[str, Any],
) -> PokeNoticeEvent:
    """解析戳一戳通知事件。

    Args:
        data: 原始事件 JSON 数据。
        base_kwargs: 公共字段字典。

    Returns:
        PokeNoticeEvent 实例。
    """
    poke_text = _parse_poke_text(data)

    return PokeNoticeEvent(
        **base_kwargs,
        notice_type=data.get("notice_type", "notify"),
        group_id=data.get("group_id", 0),
        user_id=data.get("user_id", 0),
        target_id=data.get("target_id", 0),
        poke_text=poke_text,
    )
