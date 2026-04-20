"""全局防抖 + 单线程 FIFO 请求队列。

提供 RequestQueue 类，实现：
1. **1秒防抖（不重置定时器）**：首条消息启动1秒定时器，期间同会话新消息合并但不延长等待。
2. **全局单线程 FIFO 队列**：同一时刻只有一个 LLM 请求在执行，其余排队。
3. **同会话自动合并**：排队中的请求如果与新到达的请求属于同一会话，自动合并消息列表。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ybot.utils.logger import get_logger

logger = get_logger("RequestQueue")

# 处理回调类型：接收一个 PendingRequest，返回 Awaitable[None]
ProcessCallback = Callable[["PendingRequest"], Awaitable[None]]


@dataclass
class QueuedMessage:
    """队列中的单条消息。

    Attributes:
        formatted_msg: 已格式化的消息文本（带发送者元信息）。
        context_data: 附加上下文数据，由调用方自由填充。
            群消息典型字段: session_key, env_header, context_msg, last_ref_id,
                           image_urls, send_func
            私聊消息典型字段: session_key, env_header, image_urls, send_func
    """

    formatted_msg: str
    context_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingRequest:
    """一个待处理的 AI 请求（可能包含多条合并消息）。

    Attributes:
        session_key: 会话标识（如 ``group_12345``、``friend_67890``）。
        messages: 按时间顺序排列的消息列表。
        process_callback: 处理回调，由 Bot 提供。
    """

    session_key: str
    messages: list[QueuedMessage]
    process_callback: ProcessCallback


class RequestQueue:
    """防抖 + 单线程 FIFO 请求队列。

    设计要点：
    - 所有操作都在同一个事件循环线程上执行，``_pending`` / ``_queue``
      的同步读写是安全的（无需额外锁）。
    - ``_flush_pending`` 由 ``loop.call_later`` 在同步上下文调用，
      它只操作同步数据结构并通过 ``_queue_event.set()`` 唤醒异步 worker。
    - ``_queue`` 使用 ``list`` 而非 ``asyncio.Queue``，以便支持遍历/合并。

    Args:
        debounce_seconds: 防抖等待时间（秒），默认 1.0。
    """

    def __init__(self, debounce_seconds: float = 1.0) -> None:
        self._debounce_seconds = debounce_seconds

        # 防抖阶段：session_key → PendingRequest
        self._pending: dict[str, PendingRequest] = {}
        # 防抖定时器句柄：session_key → TimerHandle
        self._debounce_timers: dict[str, asyncio.TimerHandle] = {}

        # 全局 FIFO 队列（等待执行）
        self._queue: list[PendingRequest] = []
        # 通知 worker 有新任务
        self._queue_event: asyncio.Event = asyncio.Event()

        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """启动 worker 协程。必须在事件循环中调用。"""
        self._queue_event = asyncio.Event()
        self._worker_task = asyncio.create_task(self._worker())
        logger.info(
            f"RequestQueue 已启动（防抖: {self._debounce_seconds}s，单线程 FIFO）"
        )

    async def stop(self) -> None:
        """停止 worker 协程并清理资源。"""
        # 取消所有防抖定时器
        for handle in self._debounce_timers.values():
            handle.cancel()
        self._debounce_timers.clear()
        self._pending.clear()

        # 取消 worker
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        self._queue.clear()
        logger.info("RequestQueue 已停止")

    def submit(
        self,
        session_key: str,
        message: QueuedMessage,
        process_callback: ProcessCallback,
    ) -> None:
        """提交一条消息到队列（带防抖，不重置定时器）。

        - 如果该 session 已有 pending request（定时器未到期），
          将消息合并到 pending，**不重置定时器**。
        - 如果该 session 没有 pending request，创建新的并启动 1 秒定时器。

        Args:
            session_key: 会话标识。
            message: 待排队的消息。
            process_callback: 处理回调。
        """
        if session_key in self._pending:
            # 合并到已有 pending（定时器不重置）
            self._pending[session_key].messages.append(message)
            logger.debug(
                f"消息合并到 pending（session={session_key}，"
                f"当前 {len(self._pending[session_key].messages)} 条）"
            )
        else:
            # 创建新 pending + 启动定时器
            self._pending[session_key] = PendingRequest(
                session_key=session_key,
                messages=[message],
                process_callback=process_callback,
            )
            loop = asyncio.get_running_loop()
            timer = loop.call_later(
                self._debounce_seconds, self._flush_pending, session_key
            )
            self._debounce_timers[session_key] = timer
            logger.debug(
                f"新 pending 已创建（session={session_key}），"
                f"{self._debounce_seconds}s 后入队"
            )

    # ---- 内部方法 ----

    def _flush_pending(self, session_key: str) -> None:
        """防抖定时器到期回调（同步上下文）。

        将 pending request 从防抖阶段移入全局 FIFO 队列。
        如果队列中已有同 session 的请求，合并消息列表。
        """
        pending = self._pending.pop(session_key, None)
        self._debounce_timers.pop(session_key, None)

        if pending is None:
            return

        self._enqueue_or_merge(pending)

    def _enqueue_or_merge(self, request: PendingRequest) -> None:
        """将请求加入队列，若队列中已有同 session 的请求则合并。

        合并策略：将新请求的所有消息追加到已有请求的消息列表末尾。
        """
        for existing in self._queue:
            if existing.session_key == request.session_key:
                existing.messages.extend(request.messages)
                logger.debug(
                    f"消息合并到队列中已有请求（session={request.session_key}，"
                    f"合并后 {len(existing.messages)} 条）"
                )
                return

        self._queue.append(request)
        self._queue_event.set()
        logger.debug(
            f"请求已入队（session={request.session_key}，"
            f"{len(request.messages)} 条消息，队列长度: {len(self._queue)}）"
        )

    async def _worker(self) -> None:
        """单线程 worker 协程，持续从队列取任务串行执行。"""
        logger.debug("Worker 协程已启动")
        while True:
            # 等待队列中有任务
            await self._queue_event.wait()

            while self._queue:
                # 取出队首（FIFO）
                request = self._queue.pop(0)
                logger.info(
                    f"开始处理请求（session={request.session_key}，"
                    f"{len(request.messages)} 条消息）"
                )
                try:
                    await request.process_callback(request)
                except Exception as e:
                    logger.error(
                        f"处理请求时发生错误（session={request.session_key}）: {e}",
                        exc_info=True,
                    )

            # 队列已空，清除事件等待下一批
            self._queue_event.clear()
