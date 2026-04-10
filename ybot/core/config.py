"""配置加载与管理模块。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


# 默认配置内容（TOML 格式）
DEFAULT_CONFIG_TOML = """\
[server]
host = "0.0.0.0"
port = 21050

[bot]
log_level = "INFO"    # DEBUG / INFO / WARNING / ERROR

[ai]
api_base = "https://api.openai.com/v1"   # API 地址（兼容 OpenAI 格式的任意服务）
api_key = ""                               # API 密钥
model = "gpt-4o-mini"                      # 模型 ID
"""


@dataclass
class ServerConfig:
    """WebSocket 服务端配置。"""

    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class BotConfig:
    """Bot 行为配置。"""

    log_level: str = "INFO"


@dataclass
class AIConfig:
    """AI 服务配置。"""

    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"


@dataclass
class Config:
    """Y-BOT 全局配置。"""

    server: ServerConfig = field(default_factory=ServerConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    ai: AIConfig = field(default_factory=AIConfig)

    @classmethod
    def load(cls, config_path: str | Path = "config/config.toml") -> Config:
        """从 TOML 文件加载配置。

        如果配置文件不存在，将自动生成默认配置文件。

        Args:
            config_path: 配置文件路径，相对于项目根目录。

        Returns:
            Config 实例。
        """
        path = Path(config_path)

        if not path.exists():
            cls._generate_default(path)

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> Config:
        """从字典构建 Config 实例。"""
        server_data = data.get("server", {})
        bot_data = data.get("bot", {})
        ai_data = data.get("ai", {})

        server = ServerConfig(
            host=server_data.get("host", "0.0.0.0"),
            port=server_data.get("port", 8080),
        )
        bot = BotConfig(
            log_level=bot_data.get("log_level", "INFO"),
        )
        ai = AIConfig(
            api_base=ai_data.get("api_base", "https://api.openai.com/v1"),
            api_key=ai_data.get("api_key", ""),
            model=ai_data.get("model", "gpt-4o-mini"),
        )

        return cls(server=server, bot=bot, ai=ai)

    @staticmethod
    def _generate_default(path: Path) -> None:
        """生成默认配置文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
