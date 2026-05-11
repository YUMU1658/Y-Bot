"""指令注册中心。

管理所有已注册指令，提供指令解析与查找功能。
"""

from __future__ import annotations

from ybot.commands.base import BaseCommand


class CommandRegistry:
    """指令注册与分发中心。

    维护一个指令名 → BaseCommand 实例的映射表，
    提供指令文本解析功能。
    """

    def __init__(self) -> None:
        self._commands: dict[str, BaseCommand] = {}

    def register(self, command: BaseCommand) -> None:
        """注册一个指令。

        Args:
            command: 指令实例。
        """
        self._commands[command.name] = command

    def get(self, name: str) -> BaseCommand | None:
        """按名称查找指令。

        Args:
            name: 指令名（不含 /）。

        Returns:
            对应的指令实例，未找到返回 None。
        """
        return self._commands.get(name)

    def parse_command(self, text: str) -> tuple[str, str] | None:
        """解析文本是否为已注册的指令。

        Args:
            text: 纯文本消息内容（已 strip）。

        Returns:
            ``(command_name, args_text)`` 元组，非指令返回 None。
        """
        text = text.strip()
        if not text.startswith("/"):
            return None
        parts = text[1:].split(maxsplit=1)
        if not parts:
            return None
        cmd_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        if cmd_name not in self._commands:
            return None
        return (cmd_name, args)
