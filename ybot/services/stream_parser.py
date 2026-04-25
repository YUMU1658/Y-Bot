"""流式 <send_msg> 增量解析器。

对 LLM SSE 流式输出的文本 delta 进行增量缓冲扫描，
检测完整的 <send_msg>...</send_msg> 闭合标签后返回 ParsedMessage。
"""

from __future__ import annotations

import re

from ybot.services.reply_parser import ParsedMessage

# 复用 reply_parser 中相同的正则模式
_SEND_MSG_PATTERN = re.compile(
    r'<send_msg(?:\s+reply_id="(\d+)")?\s*>(.*?)</send_msg>', re.DOTALL
)

_CLOSE_TAG = "</send_msg>"
_CLOSE_TAG_LEN = len(_CLOSE_TAG)


class StreamSendMsgParser:
    """流式 <send_msg> 增量解析器。

    逐块接收 LLM 流式输出的文本 delta，在缓冲区中检测完整的
    <send_msg>...</send_msg> 标签。每当检测到一个完整标签时，
    将其提取为 ParsedMessage 返回。

    用法::

        parser = StreamSendMsgParser()
        for delta in sse_stream:
            messages = parser.feed(delta)
            for msg in messages:
                await send(msg)
        full_reply = parser.get_full_response()
    """

    def __init__(self) -> None:
        self._buffer: str = ""  # 累积缓冲区
        self._scan_pos: int = 0  # 上次扫描结束位置（避免重复扫描）

    def feed(self, delta: str) -> list[ParsedMessage]:
        """喂入一段增量文本，返回本次新检测到的完整消息列表。

        返回值可能为空列表（delta 中没有新的完整标签）、
        单条消息、或多条消息（一个 delta 跨越多个标签闭合时）。

        Args:
            delta: LLM 流式输出的增量文本片段。

        Returns:
            本次检测到的完整 ParsedMessage 列表。
        """
        self._buffer += delta
        results: list[ParsedMessage] = []

        while True:
            # 从 _scan_pos 开始查找 </send_msg> 闭合标签
            close_idx = self._buffer.find(_CLOSE_TAG, self._scan_pos)
            if close_idx == -1:
                break  # 没有新的闭合标签

            # 找到闭合标签，在 buffer[0..close_end] 范围内用正则匹配
            close_end = close_idx + _CLOSE_TAG_LEN
            # 从上次扫描位置开始搜索，确保不重复匹配已处理的标签
            match = _SEND_MSG_PATTERN.search(self._buffer, self._scan_pos, close_end)

            if match:
                reply_id_str = match.group(1)  # 可能为 None
                content = match.group(2).strip()
                reply_id = int(reply_id_str) if reply_id_str else None

                # 与 reply_parser.parse_reply 保持一致的过滤逻辑：
                # 有 reply_id 时允许空内容（QQ 支持纯引用消息）
                # 无 reply_id 时过滤空内容
                if content or reply_id is not None:
                    results.append(ParsedMessage(content=content, reply_id=reply_id))

            # 无论是否匹配成功，都将扫描位置推进到闭合标签之后
            # （避免对同一个 </send_msg> 重复处理）
            self._scan_pos = close_end

        return results

    def get_full_response(self) -> str:
        """返回累积的完整响应文本。

        在流结束后调用，获取 LLM 的完整输出用于数据库存储。

        Returns:
            完整的 LLM 响应文本。
        """
        return self._buffer
