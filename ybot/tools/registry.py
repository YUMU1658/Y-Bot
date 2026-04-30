"""工具注册中心。

管理所有已注册的工具，提供 OpenAI tools 列表生成和 tool_call 分发执行。
"""

from __future__ import annotations

import json
from typing import Any

from ybot.tools.base import BaseTool, ToolContext, ToolResult
from ybot.utils.logger import get_logger

logger = get_logger("工具")


class ToolRegistry:
    """工具注册中心。

    Attributes:
        _tools: 已注册的工具映射（name → BaseTool）。
        _ws_server: WebSocket 服务端引用。
        _bot_info: Bot 信息缓存服务引用。
        _chat_log: 群聊消息日志引用。
    """

    def __init__(
        self,
        ws_server: Any,
        bot_info: Any,
        chat_log: Any,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._ws_server = ws_server
        self._bot_info = bot_info
        self._chat_log = chat_log

    def register(self, tool: BaseTool) -> None:
        """注册一个工具。

        Args:
            tool: 工具实例。

        Raises:
            ValueError: 工具名称已被注册。
        """
        if tool.name in self._tools:
            raise ValueError(f"工具名称 '{tool.name}' 已被注册")
        self._tools[tool.name] = tool
        logger.info(f"已注册工具: {tool.name}")

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """返回所有工具的 OpenAI tools 定义列表。

        Returns:
            符合 OpenAI API ``tools`` 参数格式的列表。
        """
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def has_tools(self) -> bool:
        """是否有注册的工具。"""
        return len(self._tools) > 0

    async def execute_tool_call(
        self,
        name: str,
        arguments_str: str,
        session_key: str,
    ) -> ToolResult:
        """执行指定工具调用。

        Args:
            name: 工具名称。
            arguments_str: JSON 格式的参数字符串（来自 LLM 响应）。
            session_key: 会话标识。

        Returns:
            工具执行结果。
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.warning(f"未知工具: {name}")
            return ToolResult(success=False, message=f"未知工具: {name}")

        # 解析参数
        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError as e:
            logger.error(f"工具参数解析失败 ({name}): {e}")
            return ToolResult(success=False, message=f"参数格式错误: {e}")

        # 构建上下文
        context = ToolContext(
            session_key=session_key,
            ws_server=self._ws_server,
            bot_info=self._bot_info,
            chat_log=self._chat_log,
        )

        # 执行工具
        try:
            result = await tool.execute(arguments, context)
            logger.info(f"工具 {name} 执行完成: success={result.success}")
            logger.debug(f"工具 {name} 结果: {result.message[:200]}")
            return result
        except Exception as e:
            logger.error(f"工具 {name} 执行异常: {e}")
            return ToolResult(success=False, message=f"工具执行异常: {e}")
