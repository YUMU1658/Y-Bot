"""WebSocket 服务端模块（基于 aiocqhttp）。

使用 aiocqhttp 库创建 WebSocket 服务端，
接受 OneBot 客户端连接，接收事件数据并分发。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

from aiocqhttp import CQHttp, Event as CQEvent

from ybot.utils.logger import get_logger

logger = get_logger("WS")

# 事件处理回调类型
EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


class WebSocketServer:
    """WebSocket 服务端（基于 aiocqhttp）。

    Y-BOT 作为 WS Server 监听，等待 OneBot 实现端（NapCat/Lagrange 等）
    以反向 WebSocket 模式连接。

    Attributes:
        host: 监听地址。
        port: 监听端口。
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._event_handler: EventHandler | None = None
        self._bot = CQHttp()
        self._server_task: asyncio.Task | None = None
        self._shutdown_event: asyncio.Event | None = None

        # 注册 aiocqhttp 事件处理器
        # aiocqhttp 内部已通过 asyncio.create_task 派发事件，不会死锁
        @self._bot.on_message
        async def handle_message(event: CQEvent) -> None:
            await self._dispatch_event(dict(event))

        @self._bot.on_notice
        async def handle_notice(event: CQEvent) -> None:
            await self._dispatch_event(dict(event))

        @self._bot.on_request
        async def handle_request(event: CQEvent) -> None:
            await self._dispatch_event(dict(event))

        @self._bot.on_meta_event
        async def handle_meta_event(event: CQEvent) -> None:
            await self._dispatch_event(dict(event))

    def set_event_handler(self, handler: EventHandler) -> None:
        """设置事件处理回调。

        Args:
            handler: 异步回调函数，接收原始事件数据 dict。
        """
        self._event_handler = handler

    async def start(self) -> None:
        """启动 WebSocket 服务端。"""
        # 抑制 Quart/hypercorn 的默认日志，避免与 Y-Bot 的 logger 冲突
        for name in (
            "quart.app",
            "quart.serving",
            "hypercorn.access",
            "hypercorn.error",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

        # 创建 shutdown_event 并传入 shutdown_trigger，
        # 防止 hypercorn 注册自己的信号处理器（在 Windows 上会覆盖
        # Python 默认的 KeyboardInterrupt 行为，导致 Ctrl+C 失效）。
        self._shutdown_event = asyncio.Event()
        self._server_task = asyncio.create_task(
            self._bot.run_task(
                host=self.host,
                port=self.port,
                shutdown_trigger=self._shutdown_event.wait,
            )
        )
        logger.info(f"WebSocket 服务端已启动，监听 {self.host}:{self.port}")
        logger.info(
            "OneBot 客户端请连接 ws://%s:%d/ws/（反向 WebSocket）",
            self.host,
            self.port,
        )

    async def stop(self) -> None:
        """停止 WebSocket 服务端。"""
        # 先通知 hypercorn 优雅关闭
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self._server_task is not None:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
            logger.info("WebSocket 服务端已关闭")

    async def send_api(self, action: str, params: dict) -> None:
        """向 OneBot 客户端发送 API 调用（fire-and-forget，不等待响应）。

        通过 aiocqhttp 的 call_action 发送请求。使用 create_task
        包装以保持 fire-and-forget 语义，避免阻塞调用方。

        Args:
            action: OneBot API 动作名称（如 send_group_msg）。
            params: API 调用参数。
        """
        asyncio.create_task(self._send_api_impl(action, params))

    async def _send_api_impl(self, action: str, params: dict) -> None:
        """send_api 的实际执行，在后台任务中运行。"""
        try:
            await self._bot.call_action(action, **params)
        except Exception as e:
            logger.warning(f"发送 API 调用失败: {e}")

    async def call_api(self, action: str, params: dict, timeout: float = 10.0) -> Any:
        """调用 OneBot API 并等待响应。

        Args:
            action: OneBot API 动作名称（如 get_group_info）。
            params: API 调用参数。
            timeout: 等待响应的超时时间（秒），默认 10 秒。

        Returns:
            API 响应中的 data 字段内容（类型取决于具体 API，可能是 dict 或 list）。

        Raises:
            RuntimeError: 无可用的 OneBot 客户端连接或 API 调用失败。
            TimeoutError: 等待响应超时。
        """
        try:
            result = await asyncio.wait_for(
                self._bot.call_action(action, **params),
                timeout=timeout,
            )
            # aiocqhttp 的 call_action 已经提取了 data 字段
            # 但可能返回 None（当响应中没有 data 字段时），
            # 为了兼容上层代码（如 bot_info.py 对返回值调用 .get()），
            # 将 None 转为空 dict
            if result is None:
                return {}
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"call_api({action}) 超时（{timeout}s）")
        except Exception as e:
            raise RuntimeError(f"call_api({action}) 失败: {e}") from e

    async def _dispatch_event(self, data: dict[str, Any]) -> None:
        """将事件数据转发给 Y-Bot 的事件处理器。"""
        if self._event_handler is not None:
            try:
                await self._event_handler(data)
            except Exception as e:
                logger.error(f"事件处理器异常: {e}")
