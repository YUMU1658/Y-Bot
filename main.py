"""Y-BOT 启动入口。"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ybot.core.bot import Bot
from ybot.core.config import Config


def main() -> None:
    """加载配置并启动 Y-BOT。"""
    config = Config.load("config/config.toml")
    bot = Bot(config)
    bot.run()


if __name__ == "__main__":
    main()
