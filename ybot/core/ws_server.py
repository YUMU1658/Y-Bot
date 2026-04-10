"""WebSocket 服务端模块。

使用 websockets 库创建 WebSocket 服务端，
接受 OneBot 客户端连接，接收事件数据并分发。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

import websockets
from websockets.asyncio.server import Server, ServerConnection

from ybot.utils.logger import get_logger

logger = get_logger("WS")

# 事件处理回调类型
EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


class WebSocketServer:
    """WebSocket 服务端。

    Y-BOT 作为 WS Server 监听，等待 OneBot 实现端（NapCat/Lagrange 等）
    以反向 WebSocket 模式连接。

    Attributes:
        host: 监听地址。
        port: 监听端口。
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._server: Server | None = None
        self._clients: set[ServerConnection] = set()
        self._event_handler: EventHandler | None = None

    def set_event_handler(self, handler: EventHandler) -> None:
        """设置事件处理回调。

        Args:
            handler: 异步回调函数，接收原始事件数据 dict。
        """
        self._event_handler = handler

    async def start(self) -> None:
        """启动 WebSocket 服务端。"""
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
        )
        logger.info(f"WebSocket 服务端已启动，监听 {self.host}:{self.port}")

    async def stop(self) -> None:
        """停止 WebSocket 服务端。"""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket 服务端已关闭")

    async def send_api(self, action: str, params: dict) -> None:
        """向 OneBot 客户端发送 API 调用。

        通过已连接的 WebSocket 向所有 OneBot 客户端广播 action 请求。

        Args:
            action: OneBot API 动作名称（如 send_group_msg）。
            params: API 调用参数。
        """
        payload = json.dumps({"action": action, "params": params})
        for client in list(self._clients):
            try:
                await client.send(payload)
            except Exception as e:
                logger.warning(f"发送 API 调用失败: {e}")

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """处理单个 WebSocket 连接。"""
        remote = websocket.remote_address
        remote_str = f"{remote[0]}:{remote[1]}" if remote else "unknown"

        self._clients.add(websocket)
        logger.info(f"OneBot 客户端已连接: {remote_str}")

        try:
            async for raw_message in websocket:
                await self._handle_message(raw_message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"OneBot 客户端断开连接: {remote_str} (code={e.code})")
        except Exception as e:
            logger.error(f"处理连接时发生错误: {remote_str} - {e}")
        finally:
            self._clients.discard(websocket)
            logger.debug(
                f"已清理客户端连接: {remote_str}，当前连接数: {len(self._clients)}"
            )

    async def _handle_message(self, raw_message: str | bytes) -> None:
        """处理接收到的 WebSocket 消息。"""
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")

            data: dict[str, Any] = json.loads(raw_message)

            if self._event_handler is not None:
                await self._event_handler(data)

        except json.JSONDecodeError:
            logger.warning(f"收到无效的 JSON 数据: {raw_message[:200]}")
        except Exception as e:
            logger.error(f"处理消息时发生错误: {e}")
