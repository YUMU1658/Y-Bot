"""图片处理工具模块。

提供 GIF 检测、下载与抽帧功能，用于将 GIF 动图转换为
多帧静态图片以供 Vision 模型分析。
"""

from __future__ import annotations

import base64
from io import BytesIO

import aiohttp
from PIL import Image

from ybot.utils.logger import get_logger

logger = get_logger("ImageUtils")

# GIF 文件魔数
_GIF_MAGIC = (b"GIF87a", b"GIF89a")

# 下载大小上限（10 MB）
_MAX_DOWNLOAD_SIZE = 10 * 1024 * 1024


async def process_image_url(
    session: aiohttp.ClientSession,
    url: str,
    max_gif_frames: int = 4,
) -> list[dict]:
    """处理单个图片 URL，返回 OpenAI Vision 格式的 content items。

    - 非 GIF 图片：返回 ``[{"type": "image_url", "image_url": {"url": url}}]``
    - GIF 动图：下载 → 抽帧 → 返回多个 base64 PNG ``image_url`` items
    - 失败时返回空列表（优雅降级）

    Args:
        session: aiohttp 异步 HTTP 会话。
        url: 图片 URL。
        max_gif_frames: GIF 最大抽帧数。

    Returns:
        OpenAI Vision 格式的 content item 列表。
    """
    try:
        is_gif = await _is_gif(session, url)
    except Exception:
        logger.warning(f"GIF 检测失败，按普通图片处理: {url}")
        is_gif = False

    if not is_gif:
        return [{"type": "image_url", "image_url": {"url": url}}]

    # GIF 处理：下载 → 抽帧
    try:
        data = await _download(session, url)
    except Exception:
        logger.warning(f"GIF 下载失败，跳过: {url}")
        return []

    try:
        frames = _extract_gif_frames(data, max_frames=max_gif_frames)
    except Exception:
        logger.warning(f"GIF 抽帧失败，按普通图片处理: {url}")
        return [{"type": "image_url", "image_url": {"url": url}}]

    if not frames:
        # 抽帧结果为空（不应发生），回退到直接使用 URL
        return [{"type": "image_url", "image_url": {"url": url}}]

    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        for b64 in frames
    ]


async def _is_gif(session: aiohttp.ClientSession, url: str) -> bool:
    """通过 HEAD 请求或 URL 特征判断是否为 GIF。

    检测策略（按优先级）：
    1. HEAD 请求的 Content-Type 包含 ``image/gif``
    2. URL 路径以 ``.gif`` 结尾

    如果 HEAD 请求失败，则下载前 6 字节检查 GIF 魔数。

    Args:
        session: aiohttp 异步 HTTP 会话。
        url: 图片 URL。

    Returns:
        是否为 GIF 图片。
    """
    # 先尝试 HEAD 请求
    try:
        async with session.head(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "image/gif" in content_type.lower():
                    return True
                # Content-Type 明确是其他图片类型，则不是 GIF
                if content_type.lower().startswith("image/") and "gif" not in content_type.lower():
                    return False
    except Exception:
        pass  # HEAD 不可用，继续其他检测方式

    # URL 后缀检测
    url_lower = url.lower().split("?")[0].split("#")[0]
    if url_lower.endswith(".gif"):
        return True

    # 最后手段：下载前 6 字节检查魔数
    try:
        async with session.get(
            url,
            headers={"Range": "bytes=0-5"},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status in (200, 206):
                header_bytes = await resp.content.read(6)
                return header_bytes.startswith(_GIF_MAGIC[0]) or header_bytes.startswith(_GIF_MAGIC[1])
    except Exception:
        pass

    return False


async def _download(session: aiohttp.ClientSession, url: str) -> bytes:
    """下载图片数据，带大小限制。

    Args:
        session: aiohttp 异步 HTTP 会话。
        url: 图片 URL。

    Returns:
        图片二进制数据。

    Raises:
        ValueError: 文件大小超过限制。
        aiohttp.ClientError: 网络错误。
    """
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()

        # 检查 Content-Length（如果有）
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > _MAX_DOWNLOAD_SIZE:
            raise ValueError(
                f"GIF 文件过大: {int(content_length)} bytes > {_MAX_DOWNLOAD_SIZE} bytes"
            )

        # 流式读取，带大小限制
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.content.iter_chunked(64 * 1024):
            total += len(chunk)
            if total > _MAX_DOWNLOAD_SIZE:
                raise ValueError(
                    f"GIF 下载超过大小限制: > {_MAX_DOWNLOAD_SIZE} bytes"
                )
            chunks.append(chunk)

        return b"".join(chunks)


def _extract_gif_frames(
    data: bytes,
    max_frames: int = 4,
) -> list[str]:
    """从 GIF 二进制数据中均匀抽帧，返回 base64 PNG 字符串列表。

    如果图片不是动图（只有 1 帧），则返回该单帧。
    如果总帧数 <= max_frames，则返回全部帧。
    否则均匀抽取 max_frames 帧。

    Args:
        data: GIF 二进制数据。
        max_frames: 最大抽帧数。

    Returns:
        base64 编码的 PNG 字符串列表。
    """
    img = Image.open(BytesIO(data))

    n_frames = getattr(img, "n_frames", 1)
    is_animated = getattr(img, "is_animated", False)

    if not is_animated or n_frames <= 1:
        # 静态图片或单帧 GIF
        return [_frame_to_base64(img)]

    # 计算要抽取的帧索引（均匀分布）
    if n_frames <= max_frames:
        indices = list(range(n_frames))
    else:
        # 均匀分布，包含首尾帧
        step = (n_frames - 1) / (max_frames - 1)
        indices = [round(i * step) for i in range(max_frames)]

    frames: list[str] = []
    for idx in indices:
        img.seek(idx)
        # 将当前帧转为 RGBA 以处理透明度
        frame = img.convert("RGBA")
        frames.append(_frame_to_base64(frame))

    return frames


def _frame_to_base64(frame: Image.Image) -> str:
    """将 PIL Image 帧转为 base64 PNG 字符串。

    Args:
        frame: PIL Image 对象。

    Returns:
        base64 编码的 PNG 字符串。
    """
    buf = BytesIO()
    # 转为 RGB（PNG 不需要 alpha 时减小体积，但保留 RGBA 以支持透明）
    if frame.mode == "RGBA":
        frame.save(buf, format="PNG", optimize=True)
    else:
        frame.convert("RGB").save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
