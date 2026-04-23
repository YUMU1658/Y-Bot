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
host = "localhost"
port = 21050

[bot]
log_level = "INFO"    # DEBUG / INFO / WARNING / ERROR

[ai]
api_base = "https://api.openai.com/v1"   # API 地址（兼容 OpenAI 格式的任意服务）
api_key = ""                               # API 密钥
model = "gpt-4o-mini"                      # 模型 ID
system_prompt = ""                         # 系统提示词（可选，留空则不发送）
max_history = 20                           # 每次请求携带的最大历史消息数
context_limit = 20                         # 每次提供的参考聊天记录最大条数
context_buffer = 100                       # 内存中每个群保留的聊天记录缓冲区大小
enable_vision = false                      # 是否启用图片识别（Vision），开启后会将图片 URL 发送给 LLM
enable_cross_session = true                # 是否启用跨会话记忆共享
cross_session_max = 5                      # 跨会话引用的最大会话数
cross_session_decay = [20, 15, 10, 5, 3]   # 每个旧会话保留的消息条数（按远近递减）
"""


@dataclass
class ServerConfig:
    """WebSocket 服务端配置。"""

    host: str = "localhost"
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
    system_prompt: str = ""
    max_history: int = 20
    context_limit: int = 20  # 每次提供的参考聊天记录最大条数
    context_buffer: int = 100  # 内存中每个群保留的聊天记录缓冲区大小
    enable_vision: bool = False  # 是否启用图片识别（Vision）
    enable_cross_session: bool = True  # 是否启用跨会话记忆共享
    cross_session_max: int = 5  # 跨会话引用的最大会话数
    cross_session_decay: list[int] = field(
        default_factory=lambda: [20, 15, 10, 5, 3]
    )  # 每个旧会话保留的消息条数（按远近递减）


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
            host=server_data.get("host", "localhost"),
            port=server_data.get("port", 8080),
        )
        bot = BotConfig(
            log_level=bot_data.get("log_level", "INFO"),
        )
        ai = AIConfig(
            api_base=ai_data.get("api_base", "https://api.openai.com/v1"),
            api_key=ai_data.get("api_key", ""),
            model=ai_data.get("model", "gpt-4o-mini"),
            system_prompt=ai_data.get("system_prompt", ""),
            max_history=ai_data.get("max_history", 20),
            context_limit=ai_data.get("context_limit", 20),
            context_buffer=ai_data.get("context_buffer", 100),
            enable_vision=ai_data.get("enable_vision", False),
            enable_cross_session=ai_data.get("enable_cross_session", True),
            cross_session_max=ai_data.get("cross_session_max", 5),
            cross_session_decay=ai_data.get("cross_session_decay", [20, 15, 10, 5, 3]),
        )

        return cls(server=server, bot=bot, ai=ai)

    @staticmethod
    def _generate_default(path: Path) -> None:
        """生成默认配置文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
