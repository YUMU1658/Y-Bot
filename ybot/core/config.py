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
access_token = ""                              # WS 鉴权 Token（留空则不启用鉴权；需与 OneBot 客户端配置一致）

[bot]
log_level = "INFO"    # DEBUG / INFO / WARNING / ERROR

[ai]
api_base = "https://api.openai.com/v1"   # API 地址（兼容 OpenAI 格式的任意服务）
api_key = ""                               # API 密钥
model = "gpt-4o-mini"                      # 模型 ID
system_prompt = ""                         # 主设定/人设提示词；高级预设不应写主要人设
max_history = 20                           # 每次请求携带的最大历史消息数
context_limit = 20                         # 每次提供的参考聊天记录最大条数
context_buffer = 100                       # 内存中每个群保留的聊天记录缓冲区大小
enable_vision = false                      # 是否启用图片识别（Vision），开启后会将图片 URL 发送给 LLM
enable_stream = false                      # 是否启用流式请求（启用后 <send_msg> 标签完成即发送，无需等待完整回复）
enable_cross_session = true                # 是否启用跨会话记忆共享
cross_session_max = 5                      # 跨会话引用的最大会话数
cross_session_decay = [20, 15, 10, 5, 3]   # 每个旧会话保留的消息条数（按远近递减）
preset_enabled = true                      # 是否启用高级预设系统（关闭时仍保留最小输出协议）
preset_name = "default"                    # 预设名称，对应 config/presets/<name>.json
preset_dir = "config/presets"              # 预设 JSON 文件目录

[worldbook]
enabled = false                            # 是否启用世界书（动态知识注入系统）
worldbook_dir = "config/worldbooks"        # 世界书 JSON 文件目录
enabled_books = []                         # 仅加载指定 ID 的世界书（空=加载目录下所有已启用的）

[interceptor]
enabled = false                            # 是否启用截断器（同会话新消息到达时，用小模型判断是否打断当前回复）
api_base = ""                              # 截断器模型 API 地址（留空则复用 [ai] 的 api_base）
api_key = ""                               # 截断器模型 API 密钥（留空则复用 [ai] 的 api_key）
model = "gpt-4o-mini"                      # 截断器使用的模型（建议轻量快速模型）
timeout = 8.0                              # 截断器判断超时时间（秒），超时视为不打断

[poke]
cooldown = 10                              # 对同一用户的戳一戳冷却时间（秒），默认 10
daily_limit = 200                          # 每日戳一戳总次数上限，默认 200

[tools]
enabled = true                             # 是否启用工具调用（function calling），关闭后 LLM 不会收到 tools 参数
"""


@dataclass
class ServerConfig:
    """WebSocket 服务端配置。"""

    host: str = "localhost"
    port: int = 8080
    access_token: str = ""


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
    enable_stream: bool = False  # 是否启用流式请求
    enable_cross_session: bool = True  # 是否启用跨会话记忆共享
    cross_session_max: int = 5  # 跨会话引用的最大会话数
    cross_session_decay: list[int] = field(
        default_factory=lambda: [20, 15, 10, 5, 3]
    )  # 每个旧会话保留的消息条数（按远近递减）
    preset_enabled: bool = True  # 是否启用高级预设系统
    preset_name: str = "default"  # 预设名称，对应 config/presets/<name>.json
    preset_dir: str = "config/presets"  # 预设 JSON 文件目录


@dataclass
class InterceptorConfig:
    """截断器配置。"""

    enabled: bool = False
    api_base: str = ""  # 留空复用 AIConfig.api_base
    api_key: str = ""  # 留空复用 AIConfig.api_key
    model: str = "gpt-4o-mini"
    timeout: float = 8.0


@dataclass
class PokeConfig:
    """戳一戳配置。"""

    cooldown: int = 10  # 对同一用户的冷却时间（秒）
    daily_limit: int = 200  # 每日总次数上限


@dataclass
class WorldBookConfig:
    """世界书配置。"""

    enabled: bool = False
    worldbook_dir: str = "config/worldbooks"
    enabled_books: list[str] = field(default_factory=list)


@dataclass
class ToolsConfig:
    """工具调用配置。"""

    enabled: bool = True  # 是否启用工具调用（function calling）


@dataclass
class Config:
    """Y-BOT 全局配置。"""

    server: ServerConfig = field(default_factory=ServerConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    interceptor: InterceptorConfig = field(default_factory=InterceptorConfig)
    worldbook: WorldBookConfig = field(default_factory=WorldBookConfig)
    poke: PokeConfig = field(default_factory=PokeConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

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
        interceptor_data = data.get("interceptor", {})
        worldbook_data = data.get("worldbook", {})
        poke_data = data.get("poke", {})
        tools_data = data.get("tools", {})

        server = ServerConfig(
            host=server_data.get("host", "localhost"),
            port=server_data.get("port", 8080),
            access_token=server_data.get("access_token", ""),
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
            enable_stream=ai_data.get("enable_stream", False),
            enable_cross_session=ai_data.get("enable_cross_session", True),
            cross_session_max=ai_data.get("cross_session_max", 5),
            cross_session_decay=ai_data.get("cross_session_decay", [20, 15, 10, 5, 3]),
            preset_enabled=ai_data.get("preset_enabled", True),
            preset_name=ai_data.get("preset_name", "default"),
            preset_dir=ai_data.get("preset_dir", "config/presets"),
        )
        interceptor = InterceptorConfig(
            enabled=interceptor_data.get("enabled", False),
            api_base=interceptor_data.get("api_base", ""),
            api_key=interceptor_data.get("api_key", ""),
            model=interceptor_data.get("model", "gpt-4o-mini"),
            timeout=interceptor_data.get("timeout", 8.0),
        )
        worldbook = WorldBookConfig(
            enabled=worldbook_data.get("enabled", False),
            worldbook_dir=worldbook_data.get("worldbook_dir", "config/worldbooks"),
            enabled_books=worldbook_data.get("enabled_books", []),
        )
        poke = PokeConfig(
            cooldown=poke_data.get("cooldown", 10),
            daily_limit=poke_data.get("daily_limit", 200),
        )
        tools = ToolsConfig(
            enabled=tools_data.get("enabled", True),
        )

        return cls(server=server, bot=bot, ai=ai, interceptor=interceptor, worldbook=worldbook, poke=poke, tools=tools)

    @staticmethod
    def _generate_default(path: Path) -> None:
        """生成默认配置文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
