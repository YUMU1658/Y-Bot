"""工具层基础类型定义。

定义工具基类 ``BaseTool``、执行上下文 ``ToolContext`` 和执行结果 ``ToolResult``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ybot.core.ws_server import WebSocketServer
    from ybot.services.bot_info import BotInfoService
    from ybot.storage.chat_log import SessionChatLog


@dataclass
class ToolContext:
    """工具执行时的上下文信息。

    Attributes:
        session_key: 会话标识（如 ``group_12345``）。
        ws_server: WebSocket 服务端引用，用于调用 OneBot API。
        bot_info: Bot 信息缓存服务，用于查询身份和群成员信息。
        chat_log: 群聊消息日志，用于标记撤回。
        enable_vision: 是否启用图片识别（Vision），决定工具是否可以传递图片给 LLM。
    """

    session_key: str
    ws_server: WebSocketServer
    bot_info: BotInfoService
    chat_log: SessionChatLog
    enable_vision: bool = False


@dataclass
class ToolResult:
    """工具执行结果。

    Attributes:
        success: 是否全部成功。
        message: 返回给 LLM 的结果描述。
        image_urls: 需要传递给 LLM 的图片 URL 列表（可选）。
            当工具需要让 LLM 看到图片时填充此字段，
            URL 可以是 QQ CDN 链接（由 ``process_image_url`` 下载转 base64）。
    """

    success: bool
    message: str
    image_urls: list[str] | None = None


class BaseTool(ABC):
    """工具基类。

    所有工具必须继承此类并实现抽象方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（对应 function calling 的 function name）。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（给 LLM 看的）。"""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema 格式的参数定义。"""

    @abstractmethod
    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """执行工具，返回结果。

        Args:
            arguments: 工具调用参数（由 LLM 生成，已解析为 dict）。
            context: 执行上下文。

        Returns:
            工具执行结果。
        """

    def to_openai_tool(self) -> dict[str, Any]:
        """转换为 OpenAI tools 数组中的元素格式。

        Returns:
            符合 OpenAI API ``tools`` 参数格式的字典。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
