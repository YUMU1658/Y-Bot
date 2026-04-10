"""日志工具模块。

提供带颜色的控制台日志输出，支持通过配置文件设置日志级别。
"""

from __future__ import annotations

import logging
import os
import sys


# Windows 下启用 ANSI 转义码支持
if sys.platform == "win32":
    os.system("")  # 启用 VT100 转义序列


# ANSI 颜色代码
class _Colors:
    RESET = "\033[0m"
    GRAY = "\033[90m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD_RED = "\033[1;31m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


# 日志级别对应的颜色
_LEVEL_COLORS = {
    logging.DEBUG: _Colors.GRAY,
    logging.INFO: _Colors.GREEN,
    logging.WARNING: _Colors.YELLOW,
    logging.ERROR: _Colors.RED,
    logging.CRITICAL: _Colors.BOLD_RED,
}


class ColorFormatter(logging.Formatter):
    """带颜色的日志格式化器。

    格式: [时间] [级别] [模块] 消息内容
    """

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, _Colors.RESET)
        level = record.levelname

        # 时间部分 - 灰色
        time_str = f"{_Colors.GRAY}{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}{_Colors.RESET}"
        # 级别部分 - 对应颜色
        level_str = f"{color}{level:<7}{_Colors.RESET}"
        # 模块部分 - 青色
        name_str = f"{_Colors.CYAN}{record.name}{_Colors.RESET}"
        # 消息部分
        msg = record.getMessage()

        return f"[{time_str}] [{level_str}] [{name_str}] {msg}"


def setup_logger(name: str = "Y-BOT", level: str = "INFO") -> logging.Logger:
    """配置并返回一个带颜色输出的 Logger。

    Args:
        name: Logger 名称。
        level: 日志级别字符串（DEBUG/INFO/WARNING/ERROR）。

    Returns:
        配置好的 Logger 实例。
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(ColorFormatter())

    logger.addHandler(handler)

    # 阻止日志向上传播到 root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取一个子 Logger。

    子 Logger 会继承父 Logger 的配置。

    Args:
        name: Logger 名称，通常使用模块名。

    Returns:
        Logger 实例。
    """
    return logging.getLogger(f"Y-BOT.{name}")
