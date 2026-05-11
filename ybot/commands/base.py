"""指令层基础抽象。

定义指令上下文、执行结果和抽象基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ybot.models.event import MessageEvent


@dataclass
class CommandContext:
    """指令执行上下文。

    Attributes:
        session_key: 当前会话的 session_key。
        user_id: 发送者 QQ。
        group_id: 群号（私聊为 None）。
        message_type: 消息类型（"group" | "private"）。
        is_admin: 是否为管理员。
        raw_args: 指令后的原始参数文本（去掉指令名后的部分）。
        event: 原始事件对象。
    """

    session_key: str
    user_id: int
    group_id: int | None
    message_type: str
    is_admin: bool
    raw_args: str
    event: MessageEvent


@dataclass
class CommandResult:
    """指令执行结果。

    Attributes:
        success: 是否执行成功。
        message: 回复给用户的文本（空则不回复）。
    """

    success: bool
    message: str = ""


class BaseCommand(ABC):
    """指令抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """指令名（不含 /），如 "clear"。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """指令描述。"""

    @abstractmethod
    async def execute(self, ctx: CommandContext) -> CommandResult:
        """执行指令。

        Args:
            ctx: 指令执行上下文。

        Returns:
            指令执行结果。
        """
