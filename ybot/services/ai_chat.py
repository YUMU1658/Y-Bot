"""AI 对话服务模块。

封装 OpenAI Chat Completions API 调用，
支持任何兼容 OpenAI 格式的 API 服务。
通过 ConversationStore 实现多轮对话上下文。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from ybot.core.config import AIConfig
from ybot.services.preset import PresetManager
from ybot.services.reply_parser import ParsedMessage
from ybot.services.stream_parser import StreamSendMsgParser
from ybot.storage.conversation import ConversationStore
from ybot.utils.logger import get_logger

logger = get_logger("AI")

# 北京时间 UTC+8
_CST = timezone(timedelta(hours=8))


@dataclass
class _PreparedChat:
    """_prepare_chat() 的返回值，包含 API 调用所需的全部数据。"""

    session_key: str
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]


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
        self._preset_manager = PresetManager(
            preset_dir=config.preset_dir,
            preset_name=config.preset_name,
            enabled=config.preset_enabled,
        )

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

    # ---- 公共前置逻辑 ----

    async def _prepare_chat(
        self,
        session_key: str,
        user_message: str,
        env_header: str = "",
        last_ref_msg_id: int | None = None,
        image_urls: list[str] | None = None,
        display_name: str | None = None,
    ) -> _PreparedChat | str:
        """chat() 和 chat_stream() 的公共前置逻辑。

        执行步骤 1-3.7：存储用户消息、获取历史、构建跨会话记忆、
        更新会话元数据、构建 messages 列表和 API 请求参数。

        Args:
            session_key: 会话标识。
            user_message: 用户消息文本。
            env_header: ENV 头部文本。
            last_ref_msg_id: 参考聊天记录去重边界。
            image_urls: 图片 URL 列表。
            display_name: 会话显示名称。

        Returns:
            成功时返回 _PreparedChat；前置检查失败时返回错误提示字符串。
        """
        # 前置检查
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

        # 当前轮使用 vision 时，history 中最后一条 user 消息的图片已被替换为占位符，
        # 需要先还原为原始 multimodal content，再交给预设系统包裹文本。
        if use_vision:
            for i in range(len(history) - 1, -1, -1):
                if history[i].get("role") == "user":
                    history[i] = {"role": "user", "content": content_array}
                    break

        # 3. 构建跨会话记忆
        cross_session_message: str | None = None
        if self._config.enable_cross_session:
            other_sessions = await self._store.get_recent_other_sessions(
                current_session_key=session_key,
                max_sessions=self._config.cross_session_max,
                decay_limits=self._config.cross_session_decay,
            )
            cross_session_message = self._build_cross_session_message(other_sessions)

        # 3.6 更新唤醒时间和显示名称
        await self._store.update_session_meta(session_key, display_name)

        # 3.7 构建最终 messages 列表
        messages = self._preset_manager.build_messages(
            env_header=env_header,
            character_prompt=self._config.system_prompt,
            history=history,
            cross_session_message=cross_session_message,
        )

        # 构建 API 请求参数
        url = f"{self._config.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
        }

        return _PreparedChat(
            session_key=session_key,
            url=url,
            headers=headers,
            payload=payload,
        )

    # ---- 非流式对话 ----

    async def chat(
        self,
        session_key: str,
        user_message: str,
        env_header: str = "",
        last_ref_msg_id: int | None = None,
        image_urls: list[str] | None = None,
        display_name: str | None = None,
        on_prepared: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> str:
        """发送多轮对话请求，返回 AI 回复文本（非流式）。

        流程：
        1-3.7. 前置逻辑（由 _prepare_chat 完成）
        4. 调用 OpenAI API（非流式，等待完整响应）
        5. 将 AI 回复存入数据库
        6. 返回回复文本

        Args:
            session_key: 会话标识（如 ``friend_12345``、``group_67890`` 或 ``temp_11111_22222``）。
            user_message: 用户消息文本（已包含元信息格式化）。
            env_header: ENV 头部文本（可选），拼接到 system prompt 前面。
            last_ref_msg_id: 本轮参考聊天记录中最新一条的 message_id（用于跨轮去重）。
            image_urls: 当前触发消息中的图片 URL 列表（可选）。仅在 enable_vision=True 时生效。
            display_name: 会话显示名称（如群名、好友昵称），用于跨会话记忆标识。
            on_prepared: 可选回调，在 _prepare_chat 完成后调用，传递构建好的 messages 列表。

        Returns:
            AI 回复的文本内容。
        """
        prepared = await self._prepare_chat(
            session_key, user_message, env_header,
            last_ref_msg_id, image_urls, display_name,
        )
        # 前置检查失败时 _prepare_chat 返回错误字符串
        if isinstance(prepared, str):
            return prepared

        # 通知调用方 messages 已构建完成
        if on_prepared:
            on_prepared(prepared.payload["messages"])

        # 4. 调用 API（非流式）
        try:
            async with self._session.post(  # type: ignore[union-attr]
                prepared.url, json=prepared.payload, headers=prepared.headers
            ) as resp:
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
        await self._store.add_message(prepared.session_key, "assistant", reply)

        # 6. 返回
        return reply

    # ---- 流式对话 ----

    async def chat_stream(
        self,
        session_key: str,
        user_message: str,
        env_header: str = "",
        last_ref_msg_id: int | None = None,
        image_urls: list[str] | None = None,
        display_name: str | None = None,
        on_message: Callable[[ParsedMessage], Awaitable[None]] | None = None,
        cancel_event: asyncio.Event | None = None,
        on_partial: Callable[[str], None] | None = None,
        on_prepared: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> str:
        """流式对话请求。

        与 chat() 共享前置逻辑，但 API 调用使用 stream=True。
        每当检测到完整的 <send_msg> 块时，通过 on_message 回调通知。
        完整回复在流结束后一次性存入数据库，与非流式行为一致。

        Args:
            session_key: 会话标识。
            user_message: 用户消息文本。
            env_header: ENV 头部文本。
            last_ref_msg_id: 参考聊天记录去重边界。
            image_urls: 图片 URL 列表。
            display_name: 会话显示名称。
            on_message: 检测到完整 <send_msg> 块时的回调函数。
            cancel_event: 可选取消信号，被 set 时中止流式接收。
            on_partial: 可选回调，每次收到增量 delta 后调用，传递当前累积的完整响应。
            on_prepared: 可选回调，在 _prepare_chat 完成后调用，传递构建好的 messages 列表。

        Returns:
            完整的 AI 回复文本（用于日志/调试）。
        """
        prepared = await self._prepare_chat(
            session_key, user_message, env_header,
            last_ref_msg_id, image_urls, display_name,
        )
        if isinstance(prepared, str):
            return prepared

        # 通知调用方 messages 已构建完成
        if on_prepared:
            on_prepared(prepared.payload["messages"])

        # 启用流式模式
        prepared.payload["stream"] = True
        parser = StreamSendMsgParser()

        # 4. 流式调用 API
        try:
            async with self._session.post(  # type: ignore[union-attr]
                prepared.url, json=prepared.payload, headers=prepared.headers
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"AI API 流式请求失败 (HTTP {resp.status}): {error_text[:200]}"
                    )
                    return f"[AI 请求失败: HTTP {resp.status}]"

                async for raw_line in resp.content:
                    # 检查取消信号（在处理每行之前检查，最小化延迟）
                    if cancel_event and cancel_event.is_set():
                        logger.info("流式请求被截断器打断，中止接收")
                        break

                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

                    if not delta:
                        continue

                    # 喂入增量解析器
                    new_messages = parser.feed(delta)

                    # 通知调用方当前累积的部分回复
                    if on_partial:
                        on_partial(parser.get_full_response())

                    if on_message:
                        for msg in new_messages:
                            await on_message(msg)

        except aiohttp.ClientError as e:
            logger.error(f"AI API 流式网络错误: {e}")
            return "[AI 网络错误]"
        except Exception as e:
            logger.error(f"AI 流式调用时发生未知错误: {e}")
            return "[AI 调用失败]"

        # 获取完整回复
        reply = parser.get_full_response()

        # 被截断器打断时不存入 AI 回复（不完整的回复不应进入历史）
        cancelled = cancel_event is not None and cancel_event.is_set()
        if not cancelled:
            # 5. 存入 AI 回复
            await self._store.add_message(prepared.session_key, "assistant", reply)

        return reply

    @staticmethod
    def _build_cross_session_message(
        other_sessions: list[dict[str, Any]],
    ) -> str | None:
        """将跨会话记录格式化为注入消息。

        每个旧会话的消息内容保持原始格式（含完整元信息），
        assistant 消息保留原始格式（含 <send_msg> 标签）。

        Args:
            other_sessions: 其他会话的压缩记录列表。

        Returns:
            格式化后的跨会话记忆消息文本，如果没有旧会话则返回 None。
        """
        if not other_sessions:
            return None

        lines = [
            "=== 以下是其他会话的近期记录（仅供参考，帮助你了解自己在其他场景的交互） ==="
        ]

        for session_data in other_sessions:
            sk = session_data["session_key"]
            display_name = session_data["display_name"]
            invoked_at = session_data["last_invoked_at"]
            messages = session_data["messages"]

            # 格式化会话标识和时间
            dt = datetime.fromtimestamp(invoked_at, tz=_CST)
            time_str = dt.strftime("%Y-%m-%d %H:%M")

            lines.append("")
            lines.append(f"[会话: {display_name}({sk}) | 最后唤醒: {time_str}]")

            # 直接拼接原始消息内容
            for msg in messages:
                content = msg["content"]
                if isinstance(content, list):
                    # multimodal 消息，提取文本部分
                    text_parts = [
                        item["text"]
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)

                if msg["role"] == "assistant":
                    lines.append(f"[BOT回复]\n{content}")
                else:
                    lines.append(content)

        lines.append("")
        lines.append("=== 以上是其他会话的近期记录 ===")

        return "\n".join(lines)
