"""AI 对话服务模块。

封装 OpenAI Chat Completions API 调用，
支持任何兼容 OpenAI 格式的 API 服务。
"""

from __future__ import annotations

import aiohttp

from ybot.core.config import AIConfig
from ybot.utils.logger import get_logger

logger = get_logger("AI")


class AIChatService:
    """AI 对话服务。

    使用 aiohttp 异步调用 OpenAI 格式的 Chat Completions API，
    仅支持单轮对话（无上下文记忆）。

    Attributes:
        _config: AI 服务配置。
        _session: aiohttp 异步 HTTP 会话。
    """

    def __init__(self, config: AIConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """初始化 HTTP 会话。

        必须在异步上下文中调用（事件循环已运行）。
        """
        self._session = aiohttp.ClientSession()
        logger.info(f"AI 服务已初始化，模型: {self._config.model}")

    async def stop(self) -> None:
        """关闭 HTTP 会话。"""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("AI 服务已关闭")

    async def chat(self, user_message: str) -> str:
        """发送单轮对话请求，返回 AI 回复文本。

        Args:
            user_message: 用户消息文本。

        Returns:
            AI 回复的文本内容。
        """
        if not self._session:
            logger.error("AI 服务未初始化，请先调用 start()")
            return "[AI 服务未初始化]"

        if not self._config.api_key:
            logger.warning("未配置 API 密钥，跳过 AI 调用")
            return "[未配置 API 密钥]"

        url = f"{self._config.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": user_message}],
        }

        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"AI API 请求失败 (HTTP {resp.status}): {error_text[:200]}"
                    )
                    return f"[AI 请求失败: HTTP {resp.status}]"

                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except aiohttp.ClientError as e:
            logger.error(f"AI API 网络错误: {e}")
            return "[AI 网络错误]"
        except (KeyError, IndexError) as e:
            logger.error(f"AI API 响应格式异常: {e}")
            return "[AI 响应格式异常]"
        except Exception as e:
            logger.error(f"AI 调用时发生未知错误: {e}")
            return "[AI 调用失败]"
