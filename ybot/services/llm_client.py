"""OpenAI 兼容 API 的共享 HTTP 客户端。

统一 AIChatService 和 InterceptorService 的 HTTP 会话管理，
避免各自独立创建和管理 aiohttp.ClientSession。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from ybot.utils.logger import get_logger

logger = get_logger("LLM客户端")


class LLMClient:
    """OpenAI 兼容 API 的共享 HTTP 客户端。

    提供统一的 HTTP 会话生命周期管理和 API 调用方法。
    AIChatService 和 InterceptorService 通过注入此客户端
    共享同一个 aiohttp.ClientSession。

    Attributes:
        _session: aiohttp 异步 HTTP 会话。
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession | None:
        """获取底层 HTTP 会话（供需要直接访问的场景使用）。"""
        return self._session

    async def start(self) -> None:
        """初始化 HTTP 会话。

        必须在异步上下文中调用（事件循环已运行）。
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            logger.debug("LLM HTTP 客户端已初始化")

    async def stop(self) -> None:
        """关闭 HTTP 会话。"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("LLM HTTP 客户端已关闭")

    def is_ready(self) -> bool:
        """检查客户端是否已初始化且可用。"""
        return self._session is not None and not self._session.closed

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> aiohttp.ClientResponse:
        """发送 POST 请求（返回上下文管理器兼容的响应对象）。

        调用方应使用 ``async with`` 管理响应生命周期。

        Args:
            url: 请求 URL。
            headers: 请求头。
            payload: JSON 请求体。

        Returns:
            aiohttp.ClientResponse 对象。

        Raises:
            RuntimeError: 客户端未初始化。
            aiohttp.ClientError: 网络错误。
        """
        if not self._session:
            raise RuntimeError("LLMClient 未初始化，请先调用 start()")
        return await self._session.post(url, json=payload, headers=headers)
