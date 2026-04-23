"""Bot 核心类。

框架的核心协调者，负责初始化各模块、启动 WebSocket 服务端、
处理事件并格式化输出日志。
"""

from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from ybot import __version__
from ybot.core.config import Config
from ybot.core.request_queue import PendingRequest, QueuedMessage, RequestQueue
from ybot.core.ws_server import WebSocketServer
from ybot.models.event import (
    Event,
    GroupMessageEvent,
    MessageEvent,
    MetaEvent,
    NoticeEvent,
    PrivateMessageEvent,
    RequestEvent,
    parse_event,
)
from ybot.models.message import (
    parse_message,
    segments_to_content,
    segments_to_text,
)
from ybot.services.ai_chat import AIChatService
from ybot.services.bot_info import BotInfoService
from ybot.services.env_builder import EnvBuilder, MessageFormatter
from ybot.services.message_builder import text_to_segments
from ybot.services.reply_parser import parse_reply
from ybot.storage.chat_log import ChatLogEntry, GroupChatLog
from ybot.storage.conversation import ConversationStore
from ybot.utils.logger import get_logger, setup_logger

# 模块级 logger，用于 _send_reply 等非实例方法的日志
_logger = get_logger("Bot")


class Bot:
    """Y-BOT 核心类。

    负责协调配置、WebSocket 服务端和事件处理。

    Attributes:
        config: 全局配置。
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # 初始化主 Logger
        self._logger = setup_logger("Y-BOT", config.bot.log_level)
        self._msg_logger = get_logger("消息")
        self._meta_logger = get_logger("元事件")
        self._notice_logger = get_logger("通知")
        self._request_logger = get_logger("请求")

        # 初始化 WebSocket 服务端
        self._ws_server = WebSocketServer(
            host=config.server.host,
            port=config.server.port,
        )
        self._ws_server.set_event_handler(self._on_raw_event)

        # 初始化 AI 对话服务
        self._conv_store = ConversationStore()
        self._ai_chat = AIChatService(config.ai, self._conv_store)

        # 初始化 Bot 信息缓存、ENV 构建器、消息格式化器
        self._bot_info = BotInfoService(self._ws_server)
        self._env_builder = EnvBuilder(self._bot_info)
        self._msg_formatter = MessageFormatter(self._bot_info)

        # 初始化群聊消息日志缓冲区
        self._chat_log = GroupChatLog(buffer_size=config.ai.context_buffer)

        # 初始化防抖 + 单线程请求队列
        self._request_queue = RequestQueue(debounce_seconds=1.0)

        self._running = False
        self._bot_info_initialized = False

    def run(self) -> None:
        """启动 Bot，进入事件循环。

        此方法会阻塞直到收到退出信号（Ctrl+C）。
        """
        self._logger.info(f"Y-BOT v{__version__} 正在启动...")

        try:
            asyncio.run(self._async_run())
        except KeyboardInterrupt:
            pass  # 优雅退出，_async_run 中已处理清理

    async def _async_run(self) -> None:
        """异步主循环。"""
        self._running = True

        # 注册信号处理（仅 Unix）
        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._signal_handler)

        # 启动 AI 服务（需在事件循环中创建 aiohttp 会话）
        await self._conv_store.initialize()
        await self._ai_chat.start()
        await self._request_queue.start()
        await self._ws_server.start()

        self._logger.info(f"Y-BOT v{__version__} 已就绪，等待 OneBot 客户端连接...")

        # 保持运行直到收到退出信号
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    def _signal_handler(self) -> None:
        """处理退出信号。"""
        self._running = False

    async def shutdown(self) -> None:
        """优雅关闭 Bot。"""
        self._logger.info("正在关闭 Y-BOT...")
        await self._ws_server.stop()
        await self._request_queue.stop()
        await self._ai_chat.stop()
        await self._conv_store.close()
        self._logger.info("Y-BOT 已关闭")

    async def _on_raw_event(self, data: dict[str, Any]) -> None:
        """原始事件数据回调，解析并分发事件。"""
        try:
            event = parse_event(data)
            await self._handle_event(event)
        except Exception as e:
            self._logger.error(f"处理事件时发生错误: {e}")

    async def _handle_event(self, event: Event) -> None:
        """事件处理入口。

        根据事件类型进行日志输出和 AI 回复触发。
        """
        # 过滤 Bot 自身发送的消息，防止自回复死循环
        # 但仍收集 bot 自身的群消息到上下文缓冲区
        if isinstance(event, MessageEvent) and event.user_id == event.self_id:
            if isinstance(event, GroupMessageEvent):
                await self._collect_group_message(event, is_bot=True)
            return

        # 首次收到事件时，延迟初始化 BotInfoService（需要 OneBot 客户端已连接）
        if not self._bot_info_initialized:
            self._bot_info_initialized = True
            try:
                await self._bot_info.initialize()
            except Exception as e:
                self._logger.warning(
                    f"BotInfoService 初始化失败（将在后续请求中重试）: {e}"
                )

        match event:
            case GroupMessageEvent() as e:
                self._log_group_message(e)
                # 收集所有群消息到上下文缓冲区（包括非 @bot 的）
                await self._collect_group_message(e, is_bot=False)
                if self._is_at_me(e):
                    text = self._extract_content(e)
                    # 解析引用/回复消息的详情
                    text = await self._resolve_reply(text, e)
                    if text.strip():
                        session_key = f"group_{e.group_id}"
                        # 构建 ENV 头部
                        env_header = await self._env_builder.build_group_env(e.group_id)
                        # 格式化当前触发消息（附带发送者元信息）
                        formatted_msg = await self._msg_formatter.format_group_message(
                            e, text
                        )
                        # 构建带参考聊天记录的 user 消息
                        (
                            context_msg,
                            last_ref_id,
                        ) = await self._build_context_user_message(
                            e, formatted_msg, session_key
                        )
                        # 提取当前触发消息中的图片 URL
                        image_urls = self._extract_image_urls(e)
                        # 提取群名用于跨会话记忆标识（命中 BotInfoService 缓存）
                        group_info = await self._bot_info.get_group_info(e.group_id)
                        display_name = group_info.group_name or f"群{e.group_id}"
                        # 提交到防抖队列（不再直接 await AI 调用）
                        queued = QueuedMessage(
                            formatted_msg=formatted_msg,
                            context_data={
                                "type": "group",
                                "session_key": session_key,
                                "env_header": env_header,
                                "context_msg": context_msg,
                                "last_ref_id": last_ref_id,
                                "image_urls": image_urls,
                                "group_id": e.group_id,
                                "display_name": display_name,
                            },
                        )
                        self._request_queue.submit(
                            session_key, queued, self._process_ai_request
                        )
            case PrivateMessageEvent() as e:
                self._log_private_message(e)
                text = self._extract_content(e)
                # 解析引用/回复消息的详情
                text = await self._resolve_reply(text, e)
                if text.strip():
                    # 临时会话：sub_type 为 "group" 表示从群聊发起的临时私聊
                    if e.sub_type == "group":
                        temp_group_id = e.raw_data.get("sender", {}).get("group_id", 0)
                        session_key = f"temp_{temp_group_id}_{e.user_id}"
                        env_header = await self._env_builder.build_temp_env(
                            e.user_id, temp_group_id, e.sender.nickname
                        )
                    else:
                        session_key = f"friend_{e.user_id}"
                        env_header = await self._env_builder.build_private_env(
                            e.user_id, e.sender.nickname
                        )
                    # 格式化消息
                    formatted_msg = self._msg_formatter.format_private_message(e, text)
                    # 提取当前触发消息中的图片 URL
                    image_urls = self._extract_image_urls(e)
                    # 构建跨会话记忆的显示名称
                    display_name = e.sender.nickname or str(e.user_id)
                    if e.sub_type == "group":
                        # 临时会话：追加来源群名
                        temp_group_info = await self._bot_info.get_group_info(
                            temp_group_id
                        )
                        temp_group_name = temp_group_info.group_name or str(
                            temp_group_id
                        )
                        display_name = f"{display_name} ← {temp_group_name}"
                    # 提交到防抖队列（不再直接 await AI 调用）
                    queued = QueuedMessage(
                        formatted_msg=formatted_msg,
                        context_data={
                            "type": "private",
                            "session_key": session_key,
                            "env_header": env_header,
                            "image_urls": image_urls,
                            "user_id": e.user_id,
                            "display_name": display_name,
                        },
                    )
                    self._request_queue.submit(
                        session_key, queued, self._process_ai_request
                    )
            case MessageEvent() as e:
                # 兜底：未知消息类型
                text = segments_to_text(e.message)
                self._msg_logger.info(f"[{e.message_type}] 用户:{e.user_id} | {text}")
            case MetaEvent() as e:
                self._log_meta_event(e)
            case NoticeEvent() as e:
                self._log_notice_event(e)
                # 处理撤回通知，标记 ChatLog 中对应消息
                self._handle_recall(e)
            case RequestEvent() as e:
                self._log_request_event(e)
            case _:
                self._logger.debug(f"收到未知事件类型: {event.post_type}")

    # ---- 回复解析与分发 ----

    async def _send_reply(
        self,
        reply: str,
        *,
        send_func: Callable[[str, int | None], Awaitable[None]],
        interval: float = 1.0,
    ) -> None:
        """解析 AI 回复并按序发送消息。

        从 AI 回复中提取 <send_msg> 标签内的消息，按顺序逐条发送，
        每条消息之间间隔指定时间。

        Args:
            reply: AI 原始回复文本。
            send_func: 发送单条消息的异步函数（接受消息文本和可选的 reply_id）。
            interval: 多条消息之间的发送间隔（秒），默认 1.0。
        """
        messages = parse_reply(reply)

        if not messages:
            _logger.warning("AI 回复中未包含 <send_msg> 标签，跳过发送")
            _logger.debug(f"原始回复: {reply[:200]}")
            return

        for i, msg in enumerate(messages):
            await send_func(msg.content, msg.reply_id)
            # 最后一条消息后不需要等待
            if i < len(messages) - 1:
                await asyncio.sleep(interval)

    # ---- 请求队列回调 ----

    async def _process_ai_request(self, request: PendingRequest) -> None:
        """处理一个（可能已合并的）AI 请求。

        由 RequestQueue 的 worker 协程调用。根据消息类型（群聊/私聊）
        和消息数量（单条/多条合并）构建最终的 user message 并调用 LLM。

        Args:
            request: 包含一条或多条合并消息的待处理请求。
        """
        # 使用最后一条消息的上下文数据（最新状态）
        last_msg = request.messages[-1]
        data = last_msg.context_data
        session_key = data["session_key"]
        env_header = data["env_header"]
        msg_type = data["type"]

        # 合并所有消息的图片 URL
        all_image_urls: list[str] = []
        for m in request.messages:
            urls = m.context_data.get("image_urls", [])
            if urls:
                all_image_urls.extend(urls)

        # 构建发送函数
        if msg_type == "group":
            group_id = data["group_id"]
            send_func = lambda msg, rid=None, gid=group_id: self.send_group_msg(
                gid, msg, rid
            )
        else:
            user_id = data["user_id"]
            send_func = lambda msg, rid=None, uid=user_id: self.send_private_msg(
                uid, msg, rid
            )

        if len(request.messages) == 1:
            # ---- 单条消息：直接使用已预处理的数据 ----
            if msg_type == "group":
                user_message = data["context_msg"]
                last_ref_id = data["last_ref_id"]
            else:
                # 私聊没有 context_msg / last_ref_id
                user_message = last_msg.formatted_msg
                last_ref_id = None
        else:
            # ---- 多条消息合并 ----
            if msg_type == "group":
                # 第一条消息的 context_msg 包含完整的聊天记录引用 + 第一条格式化消息
                # 后续消息只需追加其 formatted_msg
                first_data = request.messages[0].context_data
                parts = [first_data["context_msg"]]
                for m in request.messages[1:]:
                    parts.append(m.formatted_msg)
                user_message = "\n".join(parts)
                # last_ref_id 使用最后一条消息的值
                last_ref_id = data["last_ref_id"]
            else:
                # 私聊：拼接所有 formatted_msg
                parts = [m.formatted_msg for m in request.messages]
                user_message = "\n".join(parts)
                last_ref_id = None

        # 调用 LLM
        reply = await self._ai_chat.chat(
            session_key,
            user_message,
            env_header,
            last_ref_msg_id=last_ref_id,
            image_urls=all_image_urls or None,
            display_name=data.get("display_name"),
        )
        await self._send_reply(reply, send_func=send_func)

    # ---- 消息发送 ----

    async def send_group_msg(
        self, group_id: int, message: str, reply_id: int | None = None
    ) -> None:
        """发送群聊消息。

        将文本中的 <at qq="..."/> 标签解析为 OneBot at 消息段，
        以消息段数组格式发送。当 reply_id 不为 None 时，在消息段
        数组最前面插入 reply 类型段以实现引用回复。

        Args:
            group_id: 目标群号。
            message: 消息文本（可能包含 <at> 标签）。
            reply_id: 要引用的消息 ID（可选）。
        """
        segments = text_to_segments(message)
        if reply_id is not None:
            segments.insert(0, {"type": "reply", "data": {"id": str(reply_id)}})
        await self._ws_server.send_api(
            "send_group_msg",
            {
                "group_id": group_id,
                "message": segments,
            },
        )

    async def send_private_msg(
        self, user_id: int, message: str, reply_id: int | None = None
    ) -> None:
        """发送私聊消息。

        将文本中的 <at qq="..."/> 标签解析为 OneBot at 消息段，
        以消息段数组格式发送。当 reply_id 不为 None 时，在消息段
        数组最前面插入 reply 类型段以实现引用回复。

        Args:
            user_id: 目标用户 QQ 号。
            message: 消息文本（可能包含 <at> 标签）。
            reply_id: 要引用的消息 ID（可选）。
        """
        segments = text_to_segments(message)
        if reply_id is not None:
            segments.insert(0, {"type": "reply", "data": {"id": str(reply_id)}})
        await self._ws_server.send_api(
            "send_private_msg",
            {
                "user_id": user_id,
                "message": segments,
            },
        )

    # ---- 辅助方法 ----

    async def _collect_group_message(
        self, event: GroupMessageEvent, *, is_bot: bool
    ) -> None:
        """收集群消息到上下文缓冲区。

        对所有群消息（包括非 @bot 的和 bot 自身的）调用，
        将消息元信息和内容表示存入 GroupChatLog。

        内容表示包含所有消息段的信息（文本、图片占位标记、
        表情标记等），不再仅限于纯文本。
        如果消息包含引用/回复段，会通过 API 解析被引用消息的详情。

        使用 event.sender 中的数据，避免为每条消息调用 API。

        Args:
            event: 群聊消息事件。
            is_bot: 是否为 bot 自身发送的消息。
        """
        text = self._extract_content(event)
        if not text.strip():
            return

        # 解析引用/回复消息的详情
        text = await self._resolve_reply(text, event)

        entry = ChatLogEntry(
            message_id=event.message_id,
            group_id=event.group_id,
            user_id=event.user_id,
            nickname=event.sender.nickname or str(event.user_id),
            card=event.sender.card or "",
            role=event.sender.role or "",
            level="",  # event.sender 不含 level，避免 API 调用
            title="",  # event.sender 不含 title，避免 API 调用
            is_friend=False,  # 避免 API 调用，非触发消息不需要精确值
            timestamp=event.time,
            text=text,
            is_bot=is_bot,
        )
        self._chat_log.add(entry)

    async def _build_context_user_message(
        self,
        event: GroupMessageEvent,
        formatted_msg: str,
        session_key: str,
    ) -> tuple[str, int | None]:
        """构建带参考聊天记录的 user 消息。

        从 GroupChatLog 获取参考聊天记录（去重已提供过的），
        与当前触发消息组合为完整的 user 消息。

        Args:
            event: 当前触发的群聊消息事件。
            formatted_msg: 已格式化的当前触发消息文本。
            session_key: 会话标识。

        Returns:
            (完整的 user 消息文本, 本轮参考记录中最新一条的 message_id)。
        """
        context_limit = self.config.ai.context_limit

        # 获取上一轮的去重边界
        prev_last_ref_id = await self._conv_store.get_last_ref_msg_id(session_key)

        # 获取参考聊天记录（排除当前触发消息本身）
        ref_entries = self._chat_log.get_between(
            group_id=event.group_id,
            after_msg_id=prev_last_ref_id,
            before_msg_id=event.message_id,
            limit=context_limit,
        )

        # 记录本轮参考记录中最新一条的 message_id
        last_ref_id: int | None = None
        if ref_entries:
            last_ref_id = ref_entries[-1].message_id

        # 组合参考记录 + 新消息
        context_msg = MessageFormatter.build_context_message(ref_entries, formatted_msg)
        return context_msg, last_ref_id

    @staticmethod
    def _is_at_me(event: GroupMessageEvent) -> bool:
        """检查群聊消息是否 @了机器人。

        Args:
            event: 群聊消息事件。

        Returns:
            如果消息中包含 @机器人 则返回 True。
        """
        bot_qq = str(event.self_id)
        for seg in event.message:
            if seg.type == "at" and seg.data.get("qq") == bot_qq:
                return True
        return False

    @staticmethod
    def _extract_content(event: MessageEvent) -> str:
        """从消息段中提取完整内容表示。

        保留所有消息段的信息：
        - text → 原文
        - at → <at qq="..."/>
        - 其他类型 → 结构化占位标记（如 [图片 file:xxx]）

        与旧版 _extract_text 的区别：不再丢弃非 text/at 段，
        而是通过 segment_to_content 将它们转为 LLM 可感知的标记。

        Args:
            event: 消息事件。

        Returns:
            包含所有消息段信息的内容文本。
        """
        return segments_to_content(event.message).strip()

    @staticmethod
    def _extract_image_urls(event: MessageEvent) -> list[str]:
        """从消息段中提取图片 URL 列表。

        遍历消息段，找到 ``type="image"`` 且 ``sub_type != 1``（排除自定义表情）
        的段，提取其 ``url`` 字段。

        Args:
            event: 消息事件。

        Returns:
            图片 URL 列表（已过滤空值）。
        """
        urls: list[str] = []
        for seg in event.message:
            if seg.type != "image":
                continue
            # sub_type=1 表示自定义表情（小表情 GIF），不适合 vision 分析
            sub_type = seg.data.get("sub_type")
            if sub_type is not None and str(sub_type) == "1":
                continue
            url = seg.data.get("url")
            if url:
                urls.append(url)
        return urls

    # 引用回复内容截断上限（字符数）
    _REPLY_CONTENT_MAX_CHARS = 80

    # 北京时间 UTC+8
    _CST = timezone(timedelta(hours=8))

    async def _resolve_reply(self, content: str, event: MessageEvent) -> str:
        """解析消息中的回复段，将占位符替换为被引用消息的详情。

        遍历消息段查找 reply 类型，通过 get_msg API 获取被引用消息，
        将 ``[回复:#msg_id]`` 占位符替换为包含发送者、时间、内容的完整引用。

        成功格式::

            [回复:#1234 ← 张三(12345) 14:30:15 | "原始消息内容..."]

        消息已撤回或不可用::

            [回复:#1234 ← 消息已撤回或无法获取]

        注意：当 OneBot 实现（如 NapCat）在引用已撤回消息时完全不上报
        reply 段时，此方法无法感知引用的存在——这是 OneBot 实现的固有限制。

        Args:
            content: 已提取的内容文本（包含 ``[回复:#id]`` 占位符）。
            event: 消息事件（用于遍历消息段获取 reply id）。

        Returns:
            替换占位符后的内容文本。如果没有 reply 段则原样返回。
        """
        for seg in event.message:
            if seg.type != "reply":
                continue

            reply_msg_id = seg.data.get("id")
            if not reply_msg_id:
                # OneBot 实现上报了 reply 段但没有 id，生成兜底提示
                self._logger.debug(f"reply 段缺少 id: {seg.data}")
                placeholder = "[回复:#?]"
                if placeholder in content:
                    content = content.replace(
                        placeholder, "[回复: 消息已撤回或无法获取]", 1
                    )
                continue

            placeholder = f"[回复:#{reply_msg_id}]"
            if placeholder not in content:
                # 占位符不在 content 中，可能是 segments_to_content 未生成
                # 将引用信息追加到 content 开头
                self._logger.debug(
                    f"占位符 {placeholder} 未在 content 中找到，追加到开头"
                )
                resolved = await self._fetch_reply_detail(reply_msg_id)
                content = resolved + "\n" + content
                continue

            resolved = await self._fetch_reply_detail(reply_msg_id)
            content = content.replace(placeholder, resolved, 1)

        return content

    async def _fetch_reply_detail(self, msg_id: str) -> str:
        """通过 get_msg API 获取被引用消息的详情并格式化。

        Args:
            msg_id: 被引用消息的 message_id。

        Returns:
            格式化后的引用标记文本。
        """
        try:
            data = await self._ws_server.call_api(
                "get_msg", {"message_id": int(msg_id)}, timeout=5.0
            )
        except Exception as e:
            self._logger.debug(f"获取引用消息失败 (msg_id={msg_id}): {e}")
            return f"[回复:#{msg_id} ← 消息已撤回或无法获取]"

        # 检查返回数据是否为空或无效
        if not data:
            return f"[回复:#{msg_id} ← 消息已撤回或无法获取]"

        # 提取发送者信息
        sender = data.get("sender", {})
        nickname = sender.get("nickname", "?")
        user_id = data.get("user_id", "?")

        # 时间戳格式化
        timestamp = data.get("time", 0)
        try:
            dt = datetime.fromtimestamp(timestamp, tz=self._CST)
            time_str = dt.strftime("%H:%M:%S")
        except (OSError, ValueError):
            time_str = "??:??:??"

        # 解析被引用消息的内容（使用 segments_to_content 保持格式统一）
        raw_message_data = data.get("message", [])
        if isinstance(raw_message_data, list) and raw_message_data:
            reply_segments = parse_message(raw_message_data)
            reply_content = segments_to_content(reply_segments)
        else:
            # 降级：使用 raw_message 字段
            reply_content = data.get("raw_message", "")

        # 检查解析后的内容是否为空（可能是已撤回的消息）
        if not reply_content.strip():
            return (
                f"[回复:#{msg_id} ← {nickname}({user_id}) {time_str}"
                f" | 消息已撤回或无法获取]"
            )

        # 截断
        max_chars = self._REPLY_CONTENT_MAX_CHARS
        if len(reply_content) > max_chars:
            reply_content = reply_content[:max_chars] + "..."

        # 检查该消息是否在 ChatLog 中被标记为已撤回
        try:
            is_recalled = self._chat_log.is_recalled(int(msg_id))
        except (ValueError, TypeError):
            is_recalled = False

        if is_recalled:
            return (
                f"[回复:#{msg_id} ← {nickname}({user_id}) {time_str}"
                f' | ⚠已撤回 | "{reply_content}"]'
            )

        return (
            f'[回复:#{msg_id} ← {nickname}({user_id}) {time_str} | "{reply_content}"]'
        )

    def _log_group_message(self, event: GroupMessageEvent) -> None:
        """格式化输出群聊消息日志。"""
        text = segments_to_text(event.message)
        nickname = event.sender.card or event.sender.nickname or str(event.user_id)
        self._msg_logger.info(
            f"[群聊] 群:{event.group_id} | 用户:{event.user_id}({nickname}) | {text}"
        )

    def _log_private_message(self, event: PrivateMessageEvent) -> None:
        """格式化输出私聊消息日志。"""
        text = segments_to_text(event.message)
        nickname = event.sender.nickname or str(event.user_id)
        self._msg_logger.info(f"[私聊] 用户:{event.user_id}({nickname}) | {text}")

    def _log_meta_event(self, event: MetaEvent) -> None:
        """记录元事件日志。"""
        match event.meta_event_type:
            case "heartbeat":
                self._meta_logger.debug(f"心跳 | self_id:{event.self_id}")
            case "lifecycle":
                sub_type = event.raw_data.get("sub_type", "unknown")
                self._meta_logger.info(
                    f"生命周期 | {sub_type} | self_id:{event.self_id}"
                )
            case _:
                self._meta_logger.debug(
                    f"{event.meta_event_type} | self_id:{event.self_id}"
                )

    def _log_notice_event(self, event: NoticeEvent) -> None:
        """记录通知事件日志。"""
        extra_info = ""
        if "group_id" in event.raw_data:
            extra_info = f" | 群:{event.raw_data['group_id']}"
        if "user_id" in event.raw_data:
            extra_info += f" | 用户:{event.raw_data['user_id']}"

        self._notice_logger.info(f"{event.notice_type}{extra_info}")

    def _handle_recall(self, event: NoticeEvent) -> None:
        """处理撤回通知，标记 GroupChatLog 中对应消息为已撤回。"""
        if event.notice_type in ("group_recall", "friend_recall"):
            msg_id = event.raw_data.get("message_id")
            if msg_id is not None:
                self._chat_log.mark_recalled(int(msg_id))

    def _log_request_event(self, event: RequestEvent) -> None:
        """记录请求事件日志。"""
        extra_info = ""
        if "user_id" in event.raw_data:
            extra_info = f" | 用户:{event.raw_data['user_id']}"
        if "group_id" in event.raw_data:
            extra_info += f" | 群:{event.raw_data['group_id']}"

        self._request_logger.info(f"{event.request_type}{extra_info}")
