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
from datetime import datetime
from typing import Any

import aiohttp

from ybot.constants import TZ_CST
from ybot.core.config import AIConfig
from ybot.services.llm_client import LLMClient
from ybot.services.preset import PresetManager
from ybot.services.reply_parser import ParsedAction
from ybot.services.stream_parser import StreamActionParser
from ybot.services.worldbook import WorldBookService
from ybot.storage.conversation import ConversationStore
from ybot.tools.base import ToolResult
from ybot.utils.image_utils import process_image_url
from ybot.utils.logger import get_logger

logger = get_logger("AI")

# tool_calls 最大循环次数（防止无限循环）
_MAX_TOOL_ROUNDS = 5


@dataclass
class ChatResult:
    """AI 对话结果。

    用于替代原先的错误字符串返回模式（``return "[AI 请求失败: ...]"``），
    避免 LLM 回复以 ``[`` 开头时被误判为错误。

    Attributes:
        success: 是否成功获取到 AI 回复。
        reply: AI 回复文本（成功时为回复内容，失败时为空字符串）。
        error: 错误描述（成功时为 None）。
    """

    success: bool
    reply: str = ""
    error: str | None = None


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

    def __init__(
        self,
        config: AIConfig,
        store: ConversationStore,
        worldbook: WorldBookService | None = None,
        tool_registry: Any | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._llm_client = llm_client
        self._session: aiohttp.ClientSession | None = None  # 兼容：无 llm_client 时自管理
        self._preset_manager = PresetManager(
            preset_dir=config.preset_dir,
            preset_name=config.preset_name,
            enabled=config.preset_enabled,
        )
        self._worldbook = worldbook
        self._tool_registry = tool_registry

    async def start(self) -> None:
        """初始化 HTTP 会话。

        如果注入了 LLMClient，使用其会话；否则自行创建。
        必须在异步上下文中调用（事件循环已运行）。
        """
        if self._llm_client:
            self._session = self._llm_client.session
        else:
            self._session = aiohttp.ClientSession()
        logger.info(f"AI 服务已初始化，模型: {self._config.model}")

    async def stop(self) -> None:
        """关闭 HTTP 会话（仅在自管理模式下关闭）。"""
        if self._llm_client:
            # LLMClient 管理的会话由 LLMClient 关闭
            self._session = None
        elif self._session:
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
    ) -> _PreparedChat | ChatResult:
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
            成功时返回 _PreparedChat；前置检查失败时返回 ChatResult(success=False)。
        """
        # 前置检查
        if not self._session:
            logger.error("AI 服务未初始化，请先调用 start()")
            return ChatResult(success=False, error="AI 服务未初始化")

        if not self._config.api_key:
            logger.warning("未配置 API 密钥，跳过 AI 调用")
            return ChatResult(success=False, error="未配置 API 密钥")

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
                items = await process_image_url(self._session, url, max_gif_frames=4)
                content_array.extend(items)
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

        # 3.65 世界书扫描
        worldbook_entries = None
        if self._worldbook and self._worldbook.is_enabled():
            worldbook_entries = self._worldbook.scan_and_collect(
                current_message=user_message,
                history=history,
                session_key=session_key,
            )
            if worldbook_entries:
                logger.debug(
                    f"世界书激活 {len(worldbook_entries)} 条条目 "
                    f"(session={session_key})"
                )

        # 3.7 构建最终 messages 列表
        messages = self._preset_manager.build_messages(
            env_header=env_header,
            character_prompt=self._config.system_prompt,
            history=history,
            cross_session_message=cross_session_message,
            worldbook_entries=worldbook_entries,
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
    ) -> ChatResult:
        """发送多轮对话请求，返回 AI 回复文本（非流式）。

        流程：
        1-3.7. 前置逻辑（由 _prepare_chat 完成）
        4. 调用 OpenAI API（非流式，等待完整响应）
        4.5. 如果响应包含 tool_calls，执行工具并循环请求
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
            ChatResult 包含成功/失败状态和回复文本。
        """
        prepared = await self._prepare_chat(
            session_key, user_message, env_header,
            last_ref_msg_id, image_urls, display_name,
        )
        # 前置检查失败时 _prepare_chat 返回 ChatResult
        if isinstance(prepared, ChatResult):
            return prepared

        # 注入 tools 参数（仅当 tool_registry 存在且有注册的工具时）
        if self._tool_registry is not None and self._tool_registry.has_tools():
            prepared.payload["tools"] = self._tool_registry.get_openai_tools()

        # 通知调用方 messages 已构建完成
        if on_prepared:
            on_prepared(prepared.payload["messages"])

        # 4. 调用 API（非流式），支持 tool_calls 循环
        try:
            reply = await self._chat_with_tools(prepared)
        except aiohttp.ClientError as e:
            logger.error(f"AI API 网络错误: {e}")
            return ChatResult(success=False, error="AI 网络错误")
        except Exception as e:
            logger.error(f"AI 调用时发生未知错误: {e}")
            return ChatResult(success=False, error="AI 调用失败")

        if reply.startswith("["):
            # 内部错误消息（来自 _chat_with_tools），不存入数据库
            return ChatResult(success=False, error=reply.strip("[]"))

        # 5. 存入 AI 回复
        await self._store.add_message(prepared.session_key, "assistant", reply)

        # 6. 返回
        return ChatResult(success=True, reply=reply)

    async def _chat_with_tools(self, prepared: _PreparedChat) -> str:
        """执行 API 调用，处理 tool_calls 循环。

        当 LLM 响应包含 tool_calls 时，执行工具并将结果追加到
        messages 列表中，然后重新请求 LLM，直到 LLM 返回最终文本回复
        或达到最大循环次数。

        Args:
            prepared: 已准备好的 API 调用数据。

        Returns:
            最终的 AI 回复文本。

        Raises:
            aiohttp.ClientError: 网络错误。
        """
        messages = prepared.payload["messages"]

        for round_idx in range(_MAX_TOOL_ROUNDS):
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

            try:
                choice = data["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError) as e:
                logger.error(f"AI API 响应格式异常: {e}")
                return "[AI 响应格式异常]"

            tool_calls = message.get("tool_calls")

            if not tool_calls:
                # 无 tool_calls，提取最终回复
                reply = message.get("content", "")
                if reply is None:
                    reply = ""
                return reply

            # 有 tool_calls，需要执行工具
            if self._tool_registry is None:
                # 不应发生：有 tools 参数但没有 registry
                logger.warning("收到 tool_calls 但 tool_registry 为 None")
                reply = message.get("content", "")
                return reply if reply else "[AI 响应异常: 未预期的 tool_calls]"

            logger.info(
                f"收到 {len(tool_calls)} 个 tool_calls "
                f"(round={round_idx + 1})"
            )

            # 将 assistant 的 tool_calls 消息追加到 messages（保持原样）
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if message.get("content"):
                assistant_msg["content"] = message["content"]
            else:
                assistant_msg["content"] = None
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # 执行工具并注入结果
            tool_results: list[tuple[dict[str, Any], str]] = []
            tool_result_objects: list[ToolResult] = []
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                func_name = func.get("name", "")
                func_args = func.get("arguments", "{}")

                logger.debug(
                    f"执行工具: {func_name}(id={tc_id}, args={func_args[:100]})"
                )

                result = await self._tool_registry.execute_tool_call(
                    func_name, func_args, prepared.session_key
                )

                # 构建 tool role message
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result.message,
                }
                messages.append(tool_msg)
                tool_results.append((tc, result.message))
                tool_result_objects.append(result)

            # 将工具调用记录存入数据库
            await self._store_tool_messages(
                prepared.session_key, assistant_msg, tool_results
            )

            # 检查是否有工具返回了图片 URL，注入 multimodal user message
            await self._inject_tool_images(tool_result_objects, messages)

            # 更新 payload 中的 messages 并继续循环
            prepared.payload["messages"] = messages

        # 达到最大循环次数
        logger.warning(
            f"tool_calls 循环达到上限 ({_MAX_TOOL_ROUNDS} 轮)，"
            "强制返回最后一轮结果"
        )
        # 移除 tools 参数，强制 LLM 生成文本回复
        prepared.payload.pop("tools", None)
        async with self._session.post(  # type: ignore[union-attr]
            prepared.url, json=prepared.payload, headers=prepared.headers
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(
                    f"AI API 最终请求失败 (HTTP {resp.status}): {error_text[:200]}"
                )
                return f"[AI 请求失败: HTTP {resp.status}]"
            data = await resp.json()

        try:
            reply = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"AI API 最终响应格式异常: {e}")
            return "[AI 响应格式异常]"

        return reply or ""

    async def _store_tool_messages(
        self,
        session_key: str,
        assistant_msg: dict[str, Any],
        tool_results: list[tuple[dict[str, Any], str]],
    ) -> None:
        """将工具调用记录（assistant tool_calls + tool results）存入数据库。

        Args:
            session_key: 会话标识。
            assistant_msg: 包含 tool_calls 的 assistant 消息。
            tool_results: (tool_call_dict, result_message) 元组列表。
        """
        # 存储 assistant 的 tool_calls 消息
        tool_calls_data = json.dumps({
            "content": assistant_msg.get("content"),
            "tool_calls": assistant_msg["tool_calls"],
        }, ensure_ascii=False)
        await self._store.add_message(
            session_key, "assistant", tool_calls_data,
            content_type="tool_calls",
        )

        # 存储每个 tool result
        for tc, result_message in tool_results:
            func = tc.get("function", {})
            tool_result_data = json.dumps({
                "tool_call_id": tc.get("id", ""),
                "name": func.get("name", ""),
                "content": result_message,
            }, ensure_ascii=False)
            await self._store.add_message(
                session_key, "tool", tool_result_data,
                content_type="tool_result",
            )

    async def _execute_tools(
        self,
        tool_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        session_key: str,
        assistant_content: str | None = None,
        *,
        log_prefix: str = "",
    ) -> tuple[dict[str, Any], list[ToolResult]]:
        """执行 tool_calls 并将结果追加到 messages 列表。

        共享逻辑，供 ``_chat_with_tools`` 和 ``chat_stream`` 使用。

        Args:
            tool_calls: 累积的 tool_calls 列表。
            messages: 当前 messages 列表（就地追加）。
            session_key: 会话标识。
            assistant_content: assistant 消息的文本内容（可能为 None）。
            log_prefix: 日志前缀（如 "流式模式"）。

        Returns:
            (assistant_msg, tool_result_objects) 元组。
        """
        # 构建 assistant 消息（含 tool_calls）
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls,
        }
        messages.append(assistant_msg)

        # 执行每个 tool_call 并构建 tool result messages
        tool_results: list[tuple[dict[str, Any], str]] = []
        tool_result_objects: list[ToolResult] = []
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            func = tc.get("function", {})
            func_name = func.get("name", "")
            func_args = func.get("arguments", "{}")

            logger.debug(
                f"{log_prefix}执行工具: {func_name}(id={tc_id}, args={func_args[:100]})"
            )

            result = await self._tool_registry.execute_tool_call(  # type: ignore[union-attr]
                func_name, func_args, session_key
            )

            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result.message,
            }
            messages.append(tool_msg)
            tool_results.append((tc, result.message))
            tool_result_objects.append(result)

        # 将工具调用记录存入数据库
        await self._store_tool_messages(session_key, assistant_msg, tool_results)

        return assistant_msg, tool_result_objects

    async def _inject_tool_images(
        self,
        tool_result_objects: list[ToolResult],
        messages: list[dict[str, Any]],
    ) -> None:
        """检查工具结果中的图片 URL，注入 multimodal user message。

        共享逻辑，供 ``_chat_with_tools`` 和 ``chat_stream`` 使用。

        Args:
            tool_result_objects: 工具执行结果列表。
            messages: 当前 messages 列表（就地追加）。
        """
        all_image_urls: list[str] = []
        for result_obj in tool_result_objects:
            if result_obj.image_urls:
                all_image_urls.extend(result_obj.image_urls)

        if all_image_urls and self._config.enable_vision and self._session:
            image_content: list[dict[str, Any]] = [
                {"type": "text", "text": "[以下是工具返回的图片]"}
            ]
            for url in all_image_urls:
                items = await process_image_url(
                    self._session, url, max_gif_frames=4
                )
                image_content.extend(items)
            messages.append({"role": "user", "content": image_content})

    @staticmethod
    def _accumulate_tool_call_deltas(
        accumulated: list[dict[str, Any]], deltas: list[dict[str, Any]]
    ) -> None:
        """将 SSE delta 中的 tool_calls 增量累积到列表中。

        SSE 流式模式下，tool_calls 以增量方式出现在 delta.tool_calls 中，
        需要按 index 累积每个 tool_call 的 id、function.name、function.arguments。

        Args:
            accumulated: 累积的 tool_calls 列表（就地修改）。
            deltas: 本次 SSE delta 中的 tool_calls 增量列表。
        """
        for tc_delta in deltas:
            idx = tc_delta.get("index", 0)
            # 扩展列表以容纳新的 index
            while len(accumulated) <= idx:
                accumulated.append({
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
            # 累积字段
            if "id" in tc_delta:
                accumulated[idx]["id"] = tc_delta["id"]
            if "type" in tc_delta:
                accumulated[idx]["type"] = tc_delta["type"]
            func_delta = tc_delta.get("function", {})
            if "name" in func_delta:
                accumulated[idx]["function"]["name"] += func_delta["name"]
            if "arguments" in func_delta:
                accumulated[idx]["function"]["arguments"] += func_delta["arguments"]

    # ---- 流式对话 ----

    async def chat_stream(
        self,
        session_key: str,
        user_message: str,
        env_header: str = "",
        last_ref_msg_id: int | None = None,
        image_urls: list[str] | None = None,
        display_name: str | None = None,
        on_action: Callable[[ParsedAction], Awaitable[None]] | None = None,
        cancel_event: asyncio.Event | None = None,
        on_partial: Callable[[str], None] | None = None,
        on_prepared: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> ChatResult:
        """流式对话请求。

        与 chat() 共享前置逻辑，但 API 调用使用 stream=True。
        每当检测到完整的 <send_msg> 块时，通过 on_message 回调通知。
        完整回复在流结束后一次性存入数据库，与非流式行为一致。
        支持工具调用：当 LLM 返回 tool_calls 时，执行工具并重新请求。

        Args:
            session_key: 会话标识。
            user_message: 用户消息文本。
            env_header: ENV 头部文本。
            last_ref_msg_id: 参考聊天记录去重边界。
            image_urls: 图片 URL 列表。
            display_name: 会话显示名称。
            on_action: 检测到完整 <send_msg> 或 <poke> 块时的回调函数。
            cancel_event: 可选取消信号，被 set 时中止流式接收。
            on_partial: 可选回调，每次收到增量 delta 后调用，传递当前累积的完整响应。
            on_prepared: 可选回调，在 _prepare_chat 完成后调用，传递构建好的 messages 列表。

        Returns:
            ChatResult 包含成功/失败状态和完整的 AI 回复文本。
        """
        prepared = await self._prepare_chat(
            session_key, user_message, env_header,
            last_ref_msg_id, image_urls, display_name,
        )
        if isinstance(prepared, ChatResult):
            return prepared

        # 注入 tools 参数（与 chat() 一致）
        if self._tool_registry is not None and self._tool_registry.has_tools():
            prepared.payload["tools"] = self._tool_registry.get_openai_tools()

        # 通知调用方 messages 已构建完成
        if on_prepared:
            on_prepared(prepared.payload["messages"])

        messages = prepared.payload["messages"]
        parser = StreamActionParser()
        cancelled = False

        # 4. 流式调用 API，支持 tool_calls 循环
        try:
            for round_idx in range(_MAX_TOOL_ROUNDS):
                prepared.payload["stream"] = True
                prepared.payload["messages"] = messages

                # 本轮累积的 tool_calls 和文本内容
                accumulated_tool_calls: list[dict[str, Any]] = []
                assistant_content = ""

                # 检查取消信号（在工具轮次之间检查）
                if cancel_event and cancel_event.is_set():
                    logger.info("流式请求在工具轮次间被截断器打断")
                    cancelled = True
                    break

                async with self._session.post(  # type: ignore[union-attr]
                    prepared.url, json=prepared.payload, headers=prepared.headers
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(
                            f"AI API 流式请求失败 (HTTP {resp.status}): {error_text[:200]}"
                        )
                        return ChatResult(success=False, error=f"AI 请求失败: HTTP {resp.status}")

                    async for raw_line in resp.content:
                        # 检查取消信号
                        if cancel_event and cancel_event.is_set():
                            logger.info("流式请求被截断器打断，中止接收")
                            cancelled = True
                            break

                        line = raw_line.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

                        # 处理 content delta
                        content = delta.get("content", "")
                        if content:
                            assistant_content += content
                            new_actions = parser.feed(content)

                            if on_partial:
                                on_partial(parser.get_full_response())

                            if on_action:
                                for action in new_actions:
                                    await on_action(action)

                        # 处理 tool_calls delta
                        tc_deltas = delta.get("tool_calls")
                        if tc_deltas:
                            self._accumulate_tool_call_deltas(
                                accumulated_tool_calls, tc_deltas
                            )

                # 被取消时退出循环
                if cancelled:
                    break

                # SSE 流结束，检查是否有 tool_calls
                if not accumulated_tool_calls:
                    break  # 纯文本回复，退出循环

                # 有 tool_calls，执行工具
                if self._tool_registry is None:
                    break

                logger.info(
                    f"流式模式收到 {len(accumulated_tool_calls)} 个 tool_calls "
                    f"(round={round_idx + 1})"
                )

                # 构建 assistant 消息（含 tool_calls）
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_content or None,
                    "tool_calls": accumulated_tool_calls,
                }
                messages.append(assistant_msg)

                # 执行工具并追加 tool result
                tool_results: list[tuple[dict[str, Any], str]] = []
                tool_result_objects: list[ToolResult] = []
                for tc in accumulated_tool_calls:
                    # 检查取消信号（在工具执行之间检查）
                    if cancel_event and cancel_event.is_set():
                        logger.info("工具执行过程中被截断器打断")
                        cancelled = True
                        break

                    tc_id = tc.get("id", "")
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    func_args = func.get("arguments", "{}")

                    logger.debug(
                        f"流式模式执行工具: {func_name}(id={tc_id}, args={func_args[:100]})"
                    )

                    result = await self._tool_registry.execute_tool_call(
                        func_name, func_args, prepared.session_key
                    )

                    tool_msg: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result.message,
                    }
                    messages.append(tool_msg)
                    tool_results.append((tc, result.message))
                    tool_result_objects.append(result)

                if cancelled:
                    # 被取消时不存储不完整的工具调用记录
                    break

                # 将工具调用记录存入数据库
                await self._store_tool_messages(
                    prepared.session_key, assistant_msg, tool_results
                )

                # 检查是否有工具返回了图片 URL，注入 multimodal user message
                await self._inject_tool_images(tool_result_objects, messages)

                # 重置 parser 以准备接收下一轮的文本回复
                parser = StreamActionParser()

                # 继续循环，重新请求 LLM
            else:
                # 达到最大循环次数，移除 tools 参数强制文本回复
                logger.warning(
                    f"流式 tool_calls 循环达到上限 ({_MAX_TOOL_ROUNDS} 轮)，"
                    "强制返回最后一轮结果"
                )
                prepared.payload.pop("tools", None)
                prepared.payload["stream"] = True
                prepared.payload["messages"] = messages
                parser = StreamActionParser()

                async with self._session.post(  # type: ignore[union-attr]
                    prepared.url, json=prepared.payload, headers=prepared.headers
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(
                            f"AI API 最终流式请求失败 (HTTP {resp.status}): {error_text[:200]}"
                        )
                        return ChatResult(success=False, error=f"AI 请求失败: HTTP {resp.status}")

                    async for raw_line in resp.content:
                        if cancel_event and cancel_event.is_set():
                            cancelled = True
                            break

                        line = raw_line.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            content = chunk["choices"][0]["delta"].get("content", "")
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

                        if not content:
                            continue

                        new_actions = parser.feed(content)
                        if on_partial:
                            on_partial(parser.get_full_response())
                        if on_action:
                            for action in new_actions:
                                await on_action(action)

        except aiohttp.ClientError as e:
            logger.error(f"AI API 流式网络错误: {e}")
            return ChatResult(success=False, error="AI 网络错误")
        except Exception as e:
            logger.error(f"AI 流式调用时发生未知错误: {e}")
            return ChatResult(success=False, error="AI 调用失败")

        # 获取完整回复
        reply = parser.get_full_response()

        # 被截断器打断时不存入 AI 回复（不完整的回复不应进入历史）
        if not cancelled:
            # 5. 存入 AI 回复
            await self._store.add_message(prepared.session_key, "assistant", reply)

        return ChatResult(success=True, reply=reply)

    async def update_last_assistant_reply(
        self, session_key: str, new_content: str
    ) -> None:
        """更新数据库中最后一条 assistant 回复的内容。

        用于在 poke 标签执行后，将原始 <poke> 标签替换为执行结果文案。

        Args:
            session_key: 会话标识。
            new_content: 替换后的回复内容。
        """
        await self._store.update_last_assistant_message(session_key, new_content)

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
            dt = datetime.fromtimestamp(invoked_at, tz=TZ_CST)
            time_str = dt.strftime("%Y-%m-%d %H:%M")

            lines.append("")
            lines.append(f"[会话: {display_name}({sk}) | 最后唤醒: {time_str}]")

            # 直接拼接原始消息内容
            for msg in messages:
                role = msg.get("role", "")

                # 跳过工具调用相关的中间消息（对跨会话记忆无意义）
                if role == "tool":
                    continue
                if role == "assistant" and "tool_calls" in msg:
                    # assistant tool_calls 消息，跳过（工具调用细节不需要跨会话展示）
                    continue

                content = msg.get("content", "")
                if content is None:
                    continue
                if isinstance(content, list):
                    # multimodal 消息，提取文本部分
                    text_parts = [
                        item["text"]
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)

                if role == "assistant":
                    lines.append(f"[BOT回复]\n{content}")
                else:
                    lines.append(content)

        lines.append("")
        lines.append("=== 以上是其他会话的近期记录 ===")

        return "\n".join(lines)
