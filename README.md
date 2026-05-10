# Y-BOT

> **This project is built with AI assistance. This README is a temporary version and will be updated as the project evolves.**

Y-BOT 是一个基于 **OneBot v11** 协议的 QQ 聊天机器人框架，以 **OpenAI 兼容 LLM API** 驱动对话生成。项目的核心目标是在 OneBot 平台上追求**极高的 RP（角色扮演）能力上限**，整体以个人实验性质为主。

## 项目状态

**v0.0.0 — 早期开发阶段**

项目目前处于初期，架构和功能仍在快速迭代中。

## 核心特性

- **全异步架构** — 基于 `asyncio`，使用反向 WebSocket 接入 OneBot 实现端（NapCatQQ / Lagrange 等）
- **深度 RP 支持** — 借鉴 SillyTavern 的 Preset + World Book 体系，提供精细的 Prompt 工程能力
  - 10 个插入位的 Preset 系统，支持分组启停
  - World Book / Lorebook：关键词、正则、常驻触发，递归扫描，互斥组，Token 预算控制
- **多轮对话与跨会话记忆** — SQLite 持久化历史，跨群/好友的上下文共享与衰减压缩
- **流式响应** — 可选 SSE 流式模式，`<send_msg>` 块完成即发送
- **消息防抖与中断判定** — 1 秒防抖合并连续消息；可选二级 LLM 判断是否中断正在生成的回复
- **视觉/多模态** — 可选图片识别，GIF 逐帧分解，base64 传入 Vision 模型
- **Function Calling** — 内置 4 个工具（消息撤回、群信息查询、联系人查询、消息/转发/头像查看）
- **戳一戳交互** — 支持发送戳一戳动作，带速率限制

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 协议层 | aiocqhttp (OneBot v11 反向 WS) |
| HTTP 客户端 | aiohttp |
| 数据库 | aiosqlite (SQLite) |
| 图像处理 | Pillow |
| 配置格式 | TOML |

## 项目结构

```
Y-BOT/
├── main.py                 # 入口
├── config/
│   ├── config.toml         # 主配置
│   ├── presets/             # Prompt 预设
│   └── worldbooks/         # World Book 知识库
└── ybot/
    ├── core/               # 核心：Bot 调度、配置、WS 服务、请求队列
    ├── models/             # 数据模型：事件、消息段
    ├── services/           # 服务层：AI 对话、Preset、World Book、ENV 构建、消息发送等
    ├── storage/            # 存储：SQLite 对话历史、内存聊天日志
    ├── tools/              # Function Calling 工具集
    └── utils/              # 工具函数：日志、图像处理
```

## 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **编辑配置**
   ```bash
   # 编辑 config/config.toml，填入 LLM API 地址、密钥、模型名称等
   ```

3. **启动 Bot**
   ```bash
   python main.py
   ```

4. **连接 OneBot 实现端** — 将 NapCatQQ / Lagrange 等配置为反向 WebSocket，连接到 `ws://localhost:21050/ws/`

## 许可证

暂未确定。

---

*本文档为临时版本，由 AI 辅助生成。*
