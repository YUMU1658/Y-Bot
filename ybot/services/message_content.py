"""消息内容解析服务。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ybot.constants import TZ_CST
from ybot.core.ws_server import WebSocketServer
from ybot.models.event import MessageEvent
from ybot.models.message import (
    parse_message,
    preview_segments_to_text,
    segments_to_content,
)
from ybot.storage.chat_log import SessionChatLog
from ybot.utils.logger import get_logger

_REPLY_CONTENT_MAX_CHARS = 80
_FORWARD_PREVIEW_MAX_ITEMS = 4
_FORWARD_PREVIEW_MAX_CHARS = 20


class MessageContentResolver:
    """负责消息段内容提取和引用、转发等富内容解析。"""

    def __init__(self, ws_server: WebSocketServer, chat_log: SessionChatLog) -> None:
        self._ws_server = ws_server
        self._chat_log = chat_log
        self._logger = get_logger("消息内容")

    def extract_content(self, event: MessageEvent) -> str:
        """从消息段中提取完整内容表示。"""
        return segments_to_content(event.message).strip()

    def extract_command_text(self, event: MessageEvent) -> str:
        """从消息段中提取纯文本，用于指令检测。"""
        parts: list[str] = []
        for seg in event.message:
            if seg.type == "text":
                parts.append(seg.data.get("text", ""))
        return "".join(parts).strip()

    def extract_image_urls(self, event: MessageEvent) -> list[str]:
        """从消息段中提取图片 URL 列表。"""
        urls: list[str] = []
        for seg in event.message:
            if seg.type != "image":
                continue
            url = seg.data.get("url")
            if url:
                urls.append(url)
        return urls

    async def resolve_reply(self, content: str, event: MessageEvent) -> str:
        """解析消息中的回复段，将占位符替换为被引用消息的详情。"""
        for seg in event.message:
            if seg.type != "reply":
                continue

            reply_msg_id = seg.data.get("id")
            if not reply_msg_id:
                self._logger.debug(f"reply 段缺少 id: {seg.data}")
                placeholder = "[回复:#?]"
                if placeholder in content:
                    content = content.replace(
                        placeholder, "[回复: 消息已撤回或无法获取]", 1
                    )
                continue

            placeholder = f"[回复:#{reply_msg_id}]"
            if placeholder not in content:
                self._logger.debug(
                    f"占位符 {placeholder} 未在 content 中找到，追加到开头"
                )
                resolved = await self.fetch_reply_detail(reply_msg_id)
                content = resolved + "\n" + content
                continue

            resolved = await self.fetch_reply_detail(reply_msg_id)
            content = content.replace(placeholder, resolved, 1)

        return content

    async def fetch_reply_detail(self, msg_id: str) -> str:
        """通过 get_msg API 获取被引用消息的详情并格式化。"""
        try:
            data = await self._ws_server.call_api(
                "get_msg", {"message_id": int(msg_id)}, timeout=5.0
            )
        except Exception as e:
            self._logger.debug(f"获取引用消息失败 (msg_id={msg_id}): {e}")
            return f"[回复:#{msg_id} ← 消息已撤回或无法获取]"

        if not data:
            return f"[回复:#{msg_id} ← 消息已撤回或无法获取]"

        sender = data.get("sender", {})
        nickname = sender.get("nickname", "?")
        user_id = data.get("user_id", "?")

        timestamp = data.get("time", 0)
        try:
            dt = datetime.fromtimestamp(timestamp, tz=TZ_CST)
            time_str = dt.strftime("%H:%M:%S")
        except (OSError, ValueError):
            time_str = "??:??:??"

        raw_message_data = data.get("message", [])
        if isinstance(raw_message_data, list) and raw_message_data:
            reply_segments = parse_message(raw_message_data)
            reply_content = segments_to_content(reply_segments)
        else:
            reply_content = data.get("raw_message", "")

        if not reply_content.strip():
            return (
                f"[回复:#{msg_id} ← {nickname}({user_id}) {time_str}"
                f" | 消息已撤回或无法获取]"
            )

        if len(reply_content) > _REPLY_CONTENT_MAX_CHARS:
            reply_content = reply_content[:_REPLY_CONTENT_MAX_CHARS] + "..."

        try:
            recall_hint = self._chat_log.get_recall_hint(int(msg_id))
        except (ValueError, TypeError):
            recall_hint = ""

        if recall_hint:
            return (
                f"[回复:#{msg_id} ← {nickname}({user_id}) {time_str}"
                f' | ⚠{recall_hint} | "{reply_content}"]'
            )

        return (
            f'[回复:#{msg_id} ← {nickname}({user_id}) {time_str} | "{reply_content}"]'
        )

    async def resolve_forward(self, content: str, event: MessageEvent) -> str:
        """解析消息中的转发段，将占位符替换为转发消息预览。"""
        for seg in event.message:
            if seg.type != "forward":
                continue

            forward_id = seg.data.get("id", "")
            if not forward_id:
                continue

            placeholder = f"「转发消息:#{forward_id}」"
            if placeholder not in content:
                self._logger.debug(
                    f"占位符 {placeholder} 未在 content 中找到，追加到末尾"
                )
                resolved = await self.fetch_forward_detail(forward_id)
                content = content + "\n" + resolved
                continue

            resolved = await self.fetch_forward_detail(forward_id)
            content = content.replace(placeholder, resolved, 1)

        return content

    async def fetch_forward_detail(self, forward_id: str) -> str:
        """通过 get_forward_msg API 获取转发消息内容并格式化为预览文本。"""
        try:
            data = await self._ws_server.call_api(
                "get_forward_msg", {"id": forward_id}, timeout=5.0
            )
        except Exception as e:
            self._logger.debug(f"获取转发消息失败 (id={forward_id}): {e}")
            return "「转发消息 | 无法获取内容」"

        if not data:
            return "「转发消息 | 无法获取内容」"

        messages = data.get("messages", [])
        if not messages:
            return "「转发消息 | 无法获取内容」"

        total_count = len(messages)
        nicknames: list[str] = []
        seen_names: set[str] = set()
        is_group = False
        for node in messages:
            if not isinstance(node, dict):
                continue
            nickname, _, _ = self.extract_node_info(node)
            if nickname and nickname not in seen_names:
                seen_names.add(nickname)
                nicknames.append(nickname)
            if not is_group and node.get("message_type") == "group":
                is_group = True

        if is_group:
            title = "群聊的聊天记录"
        elif len(nicknames) == 0:
            title = "聊天记录"
        elif len(nicknames) == 1:
            title = f"{nicknames[0]}的聊天记录"
        elif len(nicknames) == 2:
            title = f"{nicknames[0]}和{nicknames[1]}的聊天记录"
        elif len(nicknames) == 3:
            title = f"{nicknames[0]}、{nicknames[1]}和{nicknames[2]}的聊天记录"
        else:
            title = f"{nicknames[0]}、{nicknames[1]}等{len(nicknames)}人的聊天记录"

        preview_lines: list[str] = []
        for node in messages[:_FORWARD_PREVIEW_MAX_ITEMS]:
            if not isinstance(node, dict):
                continue
            nickname, user_id, raw_segs = self.extract_node_info(node)
            sender_name = nickname or (str(user_id) if user_id else "?")

            if isinstance(raw_segs, list) and raw_segs:
                segs = parse_message(raw_segs)
                preview_text = preview_segments_to_text(
                    segs, max_chars=_FORWARD_PREVIEW_MAX_CHARS
                )
            else:
                preview_text = ""

            if not preview_text:
                preview_text = "..."

            preview_lines.append(f"  {sender_name}: {preview_text}")

        header = f"「转发消息 | {title} | 共{total_count}条」"
        footer = "「/转发消息」"
        body = "\n".join(preview_lines)
        return f"{header}\n{body}\n{footer}"

    @staticmethod
    def extract_node_info(node: dict[str, Any]) -> tuple[str, str, list[Any]]:
        """从转发消息的 node 中提取发送者信息和消息段。"""
        sender = node.get("sender")
        if isinstance(sender, dict) and sender:
            nickname = sender.get("card") or sender.get("nickname") or ""
            user_id = str(node.get("user_id", "")) if node.get("user_id") else ""
            raw_segs = node.get("message", [])
            if nickname or user_id or raw_segs:
                return nickname, user_id, raw_segs if isinstance(raw_segs, list) else []

        node_data = node.get("data", {}) if isinstance(node, dict) else {}
        nickname = node_data.get("nickname", "")
        user_id = str(node_data.get("user_id", "")) if node_data.get("user_id") else ""
        raw_segs = node_data.get("message", [])
        return nickname, user_id, raw_segs if isinstance(raw_segs, list) else []
