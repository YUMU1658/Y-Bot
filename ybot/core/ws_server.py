"""WebSocket 服务端模块。

使用 websockets 库创建 WebSocket 服务端，
接受 OneBot 客户端连接，接收事件数据并分发。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable
from uuid import uuid4

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
        # echo → Future 映射，用于 call_api 请求-响应关联
        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}

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
        """向 OneBot 客户端发送 API 调用（fire-and-forget，不等待响应）。

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

    async def call_api(self, action: str, params: dict, timeout: float = 10.0) -> Any:
        """调用 OneBot API 并等待响应。

        通过 echo 字段实现请求-响应关联。向第一个已连接的客户端发送请求，
        等待对应的响应返回。

        Args:
            action: OneBot API 动作名称（如 get_group_info）。
            params: API 调用参数。
            timeout: 等待响应的超时时间（秒），默认 10 秒。

        Returns:
            API 响应中的 data 字段内容（类型取决于具体 API，可能是 dict 或 list）。

        Raises:
            RuntimeError: 无可用的 OneBot 客户端连接。
            TimeoutError: 等待响应超时。
            RuntimeError: API 调用返回错误（retcode != 0）。
        """
        if not self._clients:
            raise RuntimeError("无可用的 OneBot 客户端连接")

        # 生成唯一 echo ID
        echo = uuid4().hex

        # 创建 Future 用于等待响应
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[echo] = future

        # 构建带 echo 的请求
        payload = json.dumps({"action": action, "params": params, "echo": echo})

        # 选取第一个可用客户端发送
        client = next(iter(self._clients))
        try:
            await client.send(payload)
        except Exception as e:
            self._pending_requests.pop(echo, None)
            raise RuntimeError(f"发送 API 请求失败: {e}") from e

        # 等待响应
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(echo, None)
            raise TimeoutError(f"call_api({action}) 超时（{timeout}s）")

        # 检查响应状态
        retcode = response.get("retcode", -1)
        if retcode != 0:
            status = response.get("status", "unknown")
            msg = response.get("msg", "")
            wording = response.get("wording", "")
            error_detail = wording or msg or status
            raise RuntimeError(
                f"call_api({action}) 失败: retcode={retcode}, {error_detail}"
            )

        return response.get("data", {})

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
        """处理接收到的 WebSocket 消息。

        如果消息包含 echo 字段，视为 API 响应，填入对应的 Future；
        否则视为 OneBot 事件，以后台任务方式转发给事件处理器。

        事件处理必须以 create_task 派发而非直接 await，否则当事件处理器
        内部调用 call_api() 时，会阻塞本接收循环，导致 API 响应无法被
        读取，形成死锁。
        """
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")

            data: dict[str, Any] = json.loads(raw_message)

            # 检查是否为 API 响应（包含 echo 字段）
            echo = data.get("echo")
            if echo is not None:
                future = self._pending_requests.pop(str(echo), None)
                if future is not None and not future.done():
                    future.set_result(data)
                return

            # 普通事件，以后台任务方式转发给事件处理器
            # 不能直接 await，否则事件处理中的 call_api() 会死锁
            if self._event_handler is not None:
                asyncio.create_task(self._dispatch_event(data))

        except json.JSONDecodeError:
            logger.warning(f"收到无效的 JSON 数据: {raw_message[:200]}")
        except Exception as e:
            logger.error(f"处理消息时发生错误: {e}")

    async def _dispatch_event(self, data: dict[str, Any]) -> None:
        """在后台任务中安全地调用事件处理器。"""
        try:
            await self._event_handler(data)  # type: ignore[misc]
        except Exception as e:
            logger.error(f"事件处理器异常: {e}")
