"""事件日志格式化服务。"""

from __future__ import annotations

from ybot.models.event import (
    GroupMessageEvent,
    MetaEvent,
    NoticeEvent,
    PrivateMessageEvent,
    RequestEvent,
)
from ybot.models.message import segments_to_text
from ybot.utils.logger import get_logger


class EventLogger:
    """负责将 OneBot 事件格式化输出到对应 logger。"""

    def __init__(self) -> None:
        self._msg_logger = get_logger("消息")
        self._meta_logger = get_logger("元事件")
        self._notice_logger = get_logger("通知")
        self._request_logger = get_logger("请求")

    def log_group_message(self, event: GroupMessageEvent) -> None:
        """格式化输出群聊消息日志。"""
        text = segments_to_text(event.message)
        nickname = event.sender.card or event.sender.nickname or str(event.user_id)
        self._msg_logger.info(
            f"[群聊] 群:{event.group_id} | 用户:{event.user_id}({nickname}) | {text}"
        )

    def log_private_message(self, event: PrivateMessageEvent) -> None:
        """格式化输出私聊消息日志。"""
        text = segments_to_text(event.message)
        nickname = event.sender.nickname or str(event.user_id)
        self._msg_logger.info(f"[私聊] 用户:{event.user_id}({nickname}) | {text}")

    def log_meta_event(self, event: MetaEvent) -> None:
        """记录元事件日志。"""
        match event.meta_event_type:
            case "heartbeat":
                self._meta_logger.debug(f"心跳 | self_id:{event.self_id}")
            case "lifecycle":
                sub_type = event.raw_data.get("sub_type", "unknown")
                self._meta_logger.info(
                    f"生命周期 | {sub_type} | self_id:{event.self_id}"
                )
            case _:
                self._meta_logger.debug(
                    f"{event.meta_event_type} | self_id:{event.self_id}"
                )

    def log_notice_event(self, event: NoticeEvent) -> None:
        """记录通知事件日志。"""
        extra_info = ""
        if "group_id" in event.raw_data:
            extra_info = f" | 群:{event.raw_data['group_id']}"
        if "user_id" in event.raw_data:
            extra_info += f" | 用户:{event.raw_data['user_id']}"

        self._notice_logger.info(f"{event.notice_type}{extra_info}")

    def log_request_event(self, event: RequestEvent) -> None:
        """记录请求事件日志。"""
        extra_info = ""
        if "user_id" in event.raw_data:
            extra_info = f" | 用户:{event.raw_data['user_id']}"
        if "group_id" in event.raw_data:
            extra_info += f" | 群:{event.raw_data['group_id']}"

        self._request_logger.info(f"{event.request_type}{extra_info}")
