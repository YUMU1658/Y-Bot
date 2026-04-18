"""AI 对话服务模块。

封装 OpenAI Chat Completions API 调用，
支持任何兼容 OpenAI 格式的 API 服务。
通过 ConversationStore 实现多轮对话上下文。
"""

from __future__ import annotations

import json
from typing import Any

import aiohttp

from ybot.core.config import AIConfig
from ybot.storage.conversation import ConversationStore
from ybot.utils.logger import get_logger

logger = get_logger("AI")

# 临时默认系统提示词 —— 指导模型使用 <send_msg> 格式
# TODO: 后续迁移到配置文件或专门的提示词管理模块
_DEFAULT_SYSTEM_PROMPT = """\
# 消息发送格式

你必须使用 <send_msg> 标签来发送消息。标签外的内容不会被发送，可用于思考。

格式示例：
思考过程（不会发送）
<send_msg>这是要发送的消息</send_msg>

如需发送多条消息：
<send_msg>第一条消息</send_msg>
<send_msg>第二条消息</send_msg>

规则：
- 所有要发送给用户的内容必须放在 <send_msg></send_msg> 标签内
- 标签外的文字是你的思考过程，不会被发送
- 可以发送多条消息，系统会按顺序依次发送
- 每条消息内可以换行
- 如需@某人，使用 <at qq="QQ号"/> 标签，例如：<at qq="123456"/> 你好
- 如需@全体成员，使用 <at qq="all"/>
- <at> 标签会被转换为真实的QQ@消息
"""


class AIChatService:
    """AI 对话服务。

    使用 aiohttp 异步调用 OpenAI 格式的 Chat Completions API，
    通过 ConversationStore 维护多轮对话上下文。

    Attributes:
        _config: AI 服务配置。
        _store: 对话存储层。
        _session: aiohttp 异步 HTTP 会话。
    """

    def __init__(self, config: AIConfig, store: ConversationStore) -> None:
        self._config = config
        self._store = store
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

    async def chat(
        self,
        session_key: str,
        user_message: str,
        env_header: str = "",
        last_ref_msg_id: int | None = None,
        image_urls: list[str] | None = None,
    ) -> str:
        """发送多轮对话请求，返回 AI 回复文本。

        流程：
        1. 将用户消息存入数据库
        2. 从数据库获取历史消息
        3. 构建 messages 列表（env_header + system prompt 拼接为 system message + 历史）
        4. 调用 OpenAI API
        5. 将 AI 回复存入数据库
        6. 返回回复文本

        Args:
            session_key: 会话标识（如 ``friend_12345``、``group_67890`` 或 ``temp_11111_22222``）。
            user_message: 用户消息文本（已包含元信息格式化）。
            env_header: ENV 头部文本（可选），拼接到 system prompt 前面。
            last_ref_msg_id: 本轮参考聊天记录中最新一条的 message_id（用于跨轮去重）。
            image_urls: 当前触发消息中的图片 URL 列表（可选）。仅在 enable_vision=True 时生效。

        Returns:
            AI 回复的文本内容。
        """
        if not self._session:
            logger.error("AI 服务未初始化，请先调用 start()")
            return "[AI 服务未初始化]"

        if not self._config.api_key:
            logger.warning("未配置 API 密钥，跳过 AI 调用")
            return "[未配置 API 密钥]"

        # 1. 判断是否构建 multimodal content
        use_vision = (
            self._config.enable_vision
            and image_urls is not None
            and len(image_urls) > 0
        )

        if use_vision:
            # 构建 OpenAI Vision 格式的 multimodal content 数组
            content_array: list[dict[str, Any]] = [
                {"type": "text", "text": user_message}
            ]
            for url in image_urls:
                content_array.append({"type": "image_url", "image_url": {"url": url}})
            # 存入数据库时序列化为 JSON 字符串
            await self._store.add_message(
                session_key,
                "user",
                json.dumps(content_array, ensure_ascii=False),
                last_ref_msg_id=last_ref_msg_id,
                content_type="multimodal",
            )
        else:
            # 纯文本，保持原有行为
            await self._store.add_message(
                session_key, "user", user_message, last_ref_msg_id=last_ref_msg_id
            )

        # 2. 获取历史消息
        history = await self._store.get_history(
            session_key, limit=self._config.max_history
        )

        # 3. 构建 messages 列表
        messages: list[dict[str, Any]] = []

        # 拼接 ENV 头部 + 临时默认提示词 + system prompt 为一条 system message
        full_system_prompt = ""
        if env_header:
            full_system_prompt += env_header + "\n"
        # 拼接临时默认系统提示词（消息格式指导）
        full_system_prompt += _DEFAULT_SYSTEM_PROMPT + "\n"
        if self._config.system_prompt:
            full_system_prompt += self._config.system_prompt

        messages.append({"role": "system", "content": full_system_prompt})

        messages.extend(history)

        # 当前轮使用 vision 时，history 中最后一条 user 消息的图片已被替换为占位符，
        # 需要还原为原始 multimodal content（包含真实 image_url）
        if use_vision:
            # 从后往前找到最后一条 user 消息并替换其 content
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i] = {"role": "user", "content": content_array}
                    break

        # 4. 调用 API
        url = f"{self._config.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._config.model,
            "messages": messages,
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
                reply: str = data["choices"][0]["message"]["content"]
        except aiohttp.ClientError as e:
            logger.error(f"AI API 网络错误: {e}")
            return "[AI 网络错误]"
        except (KeyError, IndexError) as e:
            logger.error(f"AI API 响应格式异常: {e}")
            return "[AI 响应格式异常]"
        except Exception as e:
            logger.error(f"AI 调用时发生未知错误: {e}")
            return "[AI 调用失败]"

        # 5. 存入 AI 回复
        await self._store.add_message(session_key, "assistant", reply)

        # 6. 返回
        return reply
