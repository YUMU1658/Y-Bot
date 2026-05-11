"""指令处理服务。"""

from __future__ import annotations

from ybot.commands import CommandRegistry
from ybot.commands.base import CommandContext, CommandResult
from ybot.core.config import CommandsConfig
from ybot.core.ws_server import WebSocketServer
from ybot.models.event import GroupMessageEvent, MessageEvent
from ybot.utils.logger import get_logger


class CommandHandler:
    """负责指令识别、权限校验、执行和结果回复。"""

    def __init__(
        self,
        registry: CommandRegistry | None,
        config: CommandsConfig,
        ws_server: WebSocketServer,
    ) -> None:
        self._registry = registry
        self._config = config
        self._ws_server = ws_server
        self._logger = get_logger("Y-BOT")

    async def try_handle(
        self,
        event: MessageEvent,
        session_key: str,
        command_text: str,
        is_at_bot: bool,
    ) -> bool | None:
        """尝试将消息作为指令处理，保持 Bot 原有 True/False/None 语义。"""
        if self._registry is None:
            return None

        parsed = self._registry.parse_command(command_text)
        if parsed is None:
            return None

        cmd_name, args = parsed

        if isinstance(event, GroupMessageEvent):
            if self._config.require_at_in_group and not is_at_bot:
                return None

        is_admin = event.user_id in self._config.admins

        if not is_admin:
            if self._config.non_admin_behavior == "block":
                return True
            return False

        command = self._registry.get(cmd_name)
        if command is None:
            return None

        group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
        ctx = CommandContext(
            session_key=session_key,
            user_id=event.user_id,
            group_id=group_id,
            message_type=event.message_type,
            is_admin=is_admin,
            raw_args=args,
            event=event,
        )

        try:
            result = await command.execute(ctx)
        except Exception as e:
            self._logger.error(f"指令 /{cmd_name} 执行失败: {e}")
            result = CommandResult(success=False, message=f"指令执行失败: {e}")

        if result.message:
            try:
                if isinstance(event, GroupMessageEvent):
                    await self._ws_server.send_api(
                        "send_group_msg",
                        {"group_id": event.group_id, "message": result.message},
                    )
                else:
                    await self._ws_server.send_api(
                        "send_private_msg",
                        {"user_id": event.user_id, "message": result.message},
                    )
            except Exception as e:
                self._logger.error(f"指令结果回复失败: {e}")

        return True
