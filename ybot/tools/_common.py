"""工具层共享辅助函数。

提取 contact_info、group_info、viewer 等工具中重复的格式化函数。
"""

from __future__ import annotations

from datetime import datetime

from ybot.constants import SEX_LABEL_MAP, TZ_CST


def format_timestamp(ts: int) -> str:
    """将 Unix 时间戳格式化为可读时间。

    Args:
        ts: Unix 时间戳（秒）。

    Returns:
        格式化后的时间字符串，如 ``"2024-01-15 14:30:00"``。
        无效时返回 ``"未知"``。
    """
    if not ts:
        return "未知"
    try:
        dt = datetime.fromtimestamp(ts, tz=TZ_CST)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError):
        return "未知"


def sex_label(raw: str) -> str:
    """性别字段转中文。

    Args:
        raw: 原始性别字符串（``"male"`` / ``"female"``）。

    Returns:
        中文性别标签，未知时返回 ``"未知"``。
    """
    return SEX_LABEL_MAP.get(raw, "未知")
