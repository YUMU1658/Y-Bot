"""AI 请求编排服务。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ybot.core.config import Config
from ybot.core.request_queue import PendingRequest, QueuedMessage, RequestQueue
from ybot.services.ai_chat import AIChatService
from ybot.services.interceptor import InterceptorService
from ybot.services.message_sender import MessageSender
from ybot.services.reply_parser import (
    ParsedAction,
    ParsedMessage,
    ParsedPoke,
    parse_reply_actions,
)
from ybot.utils.logger import get_logger

logger = get_logger("Bot")


@dataclass
class ActiveTask:
    """当前正在处理的 AI 请求的运行时状态。"""

    cancel_event: asyncio.Event
    context_messages: list[dict[str, Any]] | None = None
    partial_response: str = ""
    completed: bool = False


@dataclass
class PreparedAIRequest:
    """普通和流式 AI 请求的共享准备结果。"""

    session_key: str
    env_header: str
    msg_type: str
    user_message: str
    last_ref_id: int | None
    all_image_urls: list[str]
    send_func: Callable[[str, int | None], Awaitable[int | None]]
    poke_func: Callable[[int], Awaitable[tuple[bool, str]]]
    display_name: str | None
    active_task: ActiveTask | None


class AIRequestHandler:
    """负责 AI 请求准备、发送动作处理和截断重试。"""

    def __init__(
        self,
        config: Config,
        ai_chat: AIChatService,
        message_sender: MessageSender,
        request_queue: RequestQueue,
        interceptor: InterceptorService | None,
    ) -> None:
        self._config = config
        self._ai_chat = ai_chat
        self._message_sender = message_sender
        self._request_queue = request_queue
        self._interceptor = interceptor
        self._active_tasks: dict[str, ActiveTask] = {}

    async def process(self, request: PendingRequest) -> None:
        """处理一个（可能已合并的）AI 请求。"""
        if self._config.ai.enable_stream:
            await self._process_stream(request)
            return

        prep = self._prepare(request)
        active_task = prep.active_task

        def on_prepared(messages: list[dict[str, Any]]) -> None:
            if active_task:
                active_task.context_messages = messages

        result = await self._ai_chat.chat(
            prep.session_key,
            prep.user_message,
            prep.env_header,
            last_ref_msg_id=prep.last_ref_id,
            image_urls=prep.all_image_urls or None,
            display_name=prep.display_name,
            on_prepared=on_prepared if active_task else None,
        )

        if active_task:
            active_task.completed = True
            self._active_tasks.pop(prep.session_key, None)

            if active_task.cancel_event.is_set():
                logger.info(f"回复被打断，重新处理 (session={prep.session_key})")
                await self._reprocess_after_interrupt(request, prep.session_key)
                return

        if not result.success:
            logger.warning(f"AI 请求失败: {result.error}")
            return

        transformed_reply = await self._send_reply(
            result.reply,
            send_func=prep.send_func,
            poke_func=prep.poke_func,
            session_key=prep.session_key,
        )

        if transformed_reply != result.reply:
            await self._ai_chat.update_last_assistant_reply(
                prep.session_key, transformed_reply
            )

    async def on_interrupt_check(
        self, session_key: str, new_message: QueuedMessage
    ) -> None:
        """同会话新消息到达时调用截断器判断是否打断。"""
        if self._interceptor is None:
            return

        active = self._active_tasks.get(session_key)
        if active is None or active.completed:
            return

        decision = await self._interceptor.should_interrupt(
            character_prompt=self._config.ai.system_prompt,
            context_messages=active.context_messages or [],
            partial_response=active.partial_response or None,
            new_message=new_message.formatted_msg,
            session_key=session_key,
        )

        if active.completed:
            logger.info(
                f"截断判断返回时任务已完成，丢弃判断结果 (session={session_key})"
            )
            return

        if decision.interrupt:
            logger.info(f"截断器决定打断 (session={session_key}): {decision.reason}")
            active.cancel_event.set()
        else:
            logger.info(f"截断器决定不打断 (session={session_key}): {decision.reason}")

    def _prepare(self, request: PendingRequest) -> PreparedAIRequest:
        """提取普通和流式 AI 请求的共享准备逻辑。"""
        last_msg = request.messages[-1]
        data = last_msg.context_data
        session_key = data["session_key"]
        env_header = data["env_header"]
        msg_type = data["type"]

        all_image_urls: list[str] = []
        for m in request.messages:
            urls = m.context_data.get("image_urls", [])
            if urls:
                all_image_urls.extend(urls)

        if msg_type == "group":
            group_id = data["group_id"]
            send_func = lambda msg, rid=None, gid=group_id, sk=session_key: self.send_group_msg(
                gid, msg, rid, session_key=sk
            )
            poke_func = lambda tid, gid=group_id: self.send_poke(tid, gid)
        else:
            user_id = data["user_id"]
            send_func = lambda msg, rid=None, uid=user_id, sk=session_key: self.send_private_msg(
                uid, msg, rid, session_key=sk
            )
            poke_func = lambda tid: self.send_poke(tid, None)

        if len(request.messages) == 1:
            user_message = data["context_msg"]
            last_ref_id = data.get("last_ref_id")
        else:
            first_data = request.messages[0].context_data
            parts = [first_data["context_msg"]]
            for m in request.messages[1:]:
                parts.append(m.formatted_msg)
            user_message = "\n".join(parts)
            last_ref_id = data.get("last_ref_id")

        interrupt_hint = request.messages[0].context_data.get("interrupt_hint")
        if interrupt_hint:
            user_message = interrupt_hint + "\n\n" + user_message

        active_task: ActiveTask | None = None
        if self._interceptor:
            active_task = ActiveTask(cancel_event=asyncio.Event())
            self._active_tasks[session_key] = active_task

        return PreparedAIRequest(
            session_key=session_key,
            env_header=env_header,
            msg_type=msg_type,
            user_message=user_message,
            last_ref_id=last_ref_id,
            all_image_urls=all_image_urls,
            send_func=send_func,
            poke_func=poke_func,
            display_name=data.get("display_name"),
            active_task=active_task,
        )

    async def _process_stream(self, request: PendingRequest) -> None:
        """流式模式的 AI 请求处理。"""
        prep = self._prepare(request)
        active_task = prep.active_task

        action_queue: asyncio.Queue[ParsedAction | None] = asyncio.Queue()
        sent_messages: list[str] = []
        poke_replacements: list[tuple[str, str]] = []

        async def on_action(action: ParsedAction) -> None:
            await action_queue.put(action)

        def on_partial(full_response: str) -> None:
            if active_task:
                active_task.partial_response = full_response

        def on_prepared(messages: list[dict[str, Any]]) -> None:
            if active_task:
                active_task.context_messages = messages

        async def sender_worker() -> None:
            while True:
                action = await action_queue.get()
                if action is None:
                    break
                await asyncio.sleep(1.0)
                if active_task and active_task.cancel_event.is_set():
                    break

                if isinstance(action, ParsedMessage):
                    await prep.send_func(action.content, action.reply_id)
                    sent_messages.append(action.content)
                elif isinstance(action, ParsedPoke):
                    success, text = await prep.poke_func(action.target_id)
                    original_tag = f'<poke target="{action.target_id}"/>'
                    if success:
                        replacement = f"[💢戳一戳] {text}"
                        await self.write_poke_chat_log(
                            session_key=prep.session_key, poke_text=text
                        )
                    else:
                        replacement = f"[戳一戳失败: {text}]"
                    poke_replacements.append((original_tag, replacement))

        sender_task = asyncio.create_task(sender_worker())

        try:
            result = await self._ai_chat.chat_stream(
                prep.session_key,
                prep.user_message,
                prep.env_header,
                last_ref_msg_id=prep.last_ref_id,
                image_urls=prep.all_image_urls or None,
                display_name=prep.display_name,
                on_action=on_action,
                cancel_event=active_task.cancel_event if active_task else None,
                on_partial=on_partial if active_task else None,
                on_prepared=on_prepared if active_task else None,
            )
        finally:
            await action_queue.put(None)

        await sender_task

        if active_task:
            active_task.completed = True
            self._active_tasks.pop(prep.session_key, None)

            if active_task.cancel_event.is_set():
                logger.info(f"流式回复被打断，重新处理 (session={prep.session_key})")
                await self._reprocess_after_interrupt(
                    request, prep.session_key, sent_messages=sent_messages or None
                )
                return

        if not result.success:
            logger.warning(f"流式 AI 请求失败: {result.error}")
            return

        reply = result.reply

        if poke_replacements and reply:
            transformed_reply = reply
            for original, replacement in poke_replacements:
                transformed_reply = transformed_reply.replace(original, replacement, 1)
            await self._ai_chat.update_last_assistant_reply(
                prep.session_key, transformed_reply
            )

        if reply and not parse_reply_actions(reply):
            logger.warning("AI 回复中未包含有效标签，跳过发送")
            logger.debug(f"原始回复: {reply[:200]}")

    async def _send_reply(
        self,
        reply: str,
        *,
        send_func: Callable[[str, int | None], Awaitable[int | None]],
        poke_func: Callable[[int], Awaitable[tuple[bool, str]]],
        session_key: str | None = None,
        interval: float = 1.0,
    ) -> str:
        """解析 AI 回复并按序发送消息/执行戳一戳。"""
        actions = parse_reply_actions(reply)

        if not actions:
            logger.warning("AI 回复中未包含有效标签，跳过发送")
            logger.debug(f"原始回复: {reply[:200]}")
            return reply

        poke_replacements: list[tuple[str, str]] = []

        for i, action in enumerate(actions):
            if isinstance(action, ParsedMessage):
                await send_func(action.content, action.reply_id)
            elif isinstance(action, ParsedPoke):
                success, text = await poke_func(action.target_id)
                original_tag = f'<poke target="{action.target_id}"/>'
                if success:
                    replacement = f"[💢戳一戳] {text}"
                    if session_key is not None:
                        await self.write_poke_chat_log(
                            session_key=session_key, poke_text=text
                        )
                else:
                    replacement = f"[戳一戳失败: {text}]"
                poke_replacements.append((original_tag, replacement))

            if i < len(actions) - 1:
                await asyncio.sleep(interval)

        transformed_reply = reply
        for original, replacement in poke_replacements:
            transformed_reply = transformed_reply.replace(original, replacement, 1)

        return transformed_reply

    async def write_poke_chat_log(
        self, *, session_key: str, poke_text: str
    ) -> None:
        """将 bot 主动发起的戳一戳写入 ChatLog。"""
        await self._message_sender.write_poke_chat_log(
            session_key=session_key, poke_text=poke_text
        )

    async def _reprocess_after_interrupt(
        self,
        original_request: PendingRequest,
        session_key: str,
        sent_messages: list[str] | None = None,
    ) -> None:
        """打断后合并队列中的消息并重新处理。"""
        merged_messages = list(original_request.messages)

        pending = self._request_queue.drain_pending(session_key)
        if pending:
            merged_messages.extend(pending.messages)

        queued = self._request_queue.drain_queued(session_key)
        if queued:
            merged_messages.extend(queued.messages)

        interrupt_hint = (
            "[系统提示：你之前正在回复的消息被打断了，因为同一会话收到了新的消息。"
            "请基于最新的完整上下文重新回复。"
        )
        if sent_messages:
            sent_text = "\n".join(f"- {msg}" for msg in sent_messages)
            interrupt_hint += (
                f"\n你之前已经发送了以下消息（用户已看到）：\n{sent_text}\n"
                "请注意衔接，避免重复已发送的内容。"
            )
        interrupt_hint += "]"

        merged_messages[0].context_data["interrupt_hint"] = interrupt_hint

        new_request = PendingRequest(
            session_key=session_key,
            messages=merged_messages,
            process_callback=self.process,
        )

        logger.info(
            f"重新处理请求 (session={session_key}，{len(merged_messages)} 条消息)"
        )
        await self.process(new_request)

    async def send_group_msg(
        self,
        group_id: int,
        message: str,
        reply_id: int | None = None,
        *,
        session_key: str | None = None,
    ) -> int | None:
        """发送群聊消息。"""
        return await self._message_sender.send_group_msg(
            group_id, message, reply_id, session_key=session_key
        )

    async def send_private_msg(
        self,
        user_id: int,
        message: str,
        reply_id: int | None = None,
        *,
        session_key: str | None = None,
    ) -> int | None:
        """发送私聊消息。"""
        return await self._message_sender.send_private_msg(
            user_id, message, reply_id, session_key=session_key
        )

    async def send_poke(
        self, target_id: int, group_id: int | None = None
    ) -> tuple[bool, str]:
        """发送戳一戳。"""
        return await self._message_sender.send_poke(target_id, group_id)
