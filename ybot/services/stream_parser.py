"""流式动作增量解析器。

对 LLM SSE 流式输出的文本 delta 进行增量缓冲扫描，
检测完整的 <send_msg>...</send_msg> 和 <poke target="..."/> 标签后
返回 ParsedAction（ParsedMessage 或 ParsedPoke）。
"""

from __future__ import annotations

import re

from ybot.services.reply_parser import ParsedAction, ParsedMessage, ParsedPoke

# 复用 reply_parser 中相同的正则模式
_SEND_MSG_PATTERN = re.compile(
    r'<send_msg(?:\s+reply_id="(\d+)")?\s*>(.*?)</send_msg>', re.DOTALL
)

_POKE_PATTERN = re.compile(r'<poke\s+target="(\d+)"\s*/>')

_CLOSE_TAG = "</send_msg>"
_CLOSE_TAG_LEN = len(_CLOSE_TAG)


class StreamActionParser:
    """流式动作增量解析器。

    逐块接收 LLM 流式输出的文本 delta，在缓冲区中检测完整的
    <send_msg>...</send_msg> 和 <poke target="..."/> 标签。
    每当检测到一个完整标签时，将其提取为 ParsedAction 返回。

    动作按出现顺序返回，<send_msg> 和 <poke> 可以交错出现。

    设计要点：
    - ``_scan_pos`` 仅用于标记"从哪里开始搜索新的闭合标记"
    - 对于 ``</send_msg>``：regex 从 ``_consumed_pos`` 开始搜索
      （因为 ``<send_msg>`` 开头可能在 ``_scan_pos`` 之前）
    - 对于 ``<poke/>``：自闭合标签，regex 从 ``_scan_pos`` 开始即可
    - ``_consumed_pos`` 标记已完全处理的缓冲区位置

    用法::

        parser = StreamActionParser()
        for delta in sse_stream:
            actions = parser.feed(delta)
            for action in actions:
                if isinstance(action, ParsedMessage):
                    await send(action)
                elif isinstance(action, ParsedPoke):
                    await poke(action)
        full_reply = parser.get_full_response()
    """

    def __init__(self) -> None:
        self._buffer: str = ""  # 累积缓冲区
        self._scan_pos: int = 0  # 搜索闭合标记的起始位置
        self._consumed_pos: int = 0  # 已完全处理（消费）的缓冲区位置

    def feed(self, delta: str) -> list[ParsedAction]:
        """喂入一段增量文本，返回本次新检测到的完整动作列表。

        返回值可能为空列表（delta 中没有新的完整标签）、
        单个动作、或多个动作（一个 delta 跨越多个标签闭合时）。

        Args:
            delta: LLM 流式输出的增量文本片段。

        Returns:
            本次检测到的完整 ParsedAction 列表。
        """
        self._buffer += delta
        results: list[ParsedAction] = []

        while True:
            # 搜索 </send_msg> 闭合标签
            close_send = self._buffer.find(_CLOSE_TAG, self._scan_pos)

            # 在 [_scan_pos, close_send) 范围内搜索 <poke/> 标签
            # （poke 标签只出现在 <send_msg> 之外，即 _consumed_pos 之后）
            poke_search_end = close_send if close_send != -1 else len(self._buffer)
            poke_match = _POKE_PATTERN.search(
                self._buffer, self._consumed_pos, poke_search_end
            )

            # 决定先处理哪个（按位置排序）
            if poke_match and (close_send == -1 or poke_match.start() < close_send):
                # poke 标签在 </send_msg> 之前（或没有 </send_msg>）
                target_id = int(poke_match.group(1))
                results.append(ParsedPoke(target_id=target_id))
                self._consumed_pos = poke_match.end()
                self._scan_pos = poke_match.end()
                continue

            if close_send == -1:
                break  # 没有 </send_msg> 闭合标签

            # 找到 </send_msg>，从 _consumed_pos 开始匹配完整的 <send_msg>...</send_msg>
            close_end = close_send + _CLOSE_TAG_LEN
            match = _SEND_MSG_PATTERN.search(
                self._buffer, self._consumed_pos, close_end
            )

            if match:
                reply_id_str = match.group(1)
                content = match.group(2).strip()
                reply_id = int(reply_id_str) if reply_id_str else None

                if content or reply_id is not None:
                    results.append(
                        ParsedMessage(content=content, reply_id=reply_id)
                    )

            # 推进位置
            self._consumed_pos = close_end
            self._scan_pos = close_end

        return results

    def get_full_response(self) -> str:
        """返回累积的完整响应文本。

        在流结束后调用，获取 LLM 的完整输出用于数据库存储。

        Returns:
            完整的 LLM 响应文本。
        """
        return self._buffer


# 向后兼容别名
StreamSendMsgParser = StreamActionParser
