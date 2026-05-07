"""图片处理工具模块。

提供图片下载、base64 转换、GIF 检测与抽帧功能。
所有图片在存储时统一转为 base64，避免 QQ CDN URL 过期问题。
GIF 动图会被抽帧为多张静态图片，并附带 ``_gif_frame`` 序号标记。
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

# 常见图片格式魔数 → MIME 子类型映射
_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"RIFF", "webp"),  # WebP 以 RIFF 开头（后续还有 WEBP 标记）
    (b"BM", "bmp"),
]

# 下载大小上限（10 MB）
_MAX_DOWNLOAD_SIZE = 10 * 1024 * 1024


async def process_image_url(
    session: aiohttp.ClientSession,
    url: str,
    max_gif_frames: int = 4,
) -> list[dict]:
    """处理单个图片 URL，返回 OpenAI Vision 格式的 content items。

    所有图片都会下载并转为 base64 存储，避免 QQ CDN URL 过期问题。

    - 非 GIF 图片：下载 → base64 → 返回单个 ``image_url`` item
    - GIF 动图：下载 → 抽帧 → 返回多个 base64 PNG ``image_url`` items，
      每个 item 附带 ``_gif_frame`` 序号标记
    - 失败时优雅降级（返回原始 URL 或空列表）

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
        # 非 GIF：下载并转 base64
        try:
            data = await _download(session, url)
            b64_url = _image_to_base64(data)
            return [{"type": "image_url", "image_url": {"url": b64_url}}]
        except Exception:
            logger.warning(f"图片下载/转换失败，使用原始 URL: {url}")
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
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
            "_gif_frame": idx,
        }
        for idx, b64 in enumerate(frames)
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
                f"图片文件过大: {int(content_length)} bytes > {_MAX_DOWNLOAD_SIZE} bytes"
            )

        # 流式读取，带大小限制
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.content.iter_chunked(64 * 1024):
            total += len(chunk)
            if total > _MAX_DOWNLOAD_SIZE:
                raise ValueError(
                    f"图片下载超过大小限制: > {_MAX_DOWNLOAD_SIZE} bytes"
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


def _detect_image_mime(data: bytes) -> str:
    """通过魔数检测图片的 MIME 子类型。

    Args:
        data: 图片二进制数据。

    Returns:
        MIME 子类型字符串（如 ``"png"``、``"jpeg"``），
        检测失败时默认返回 ``"png"``。
    """
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            # WebP 需要额外检查 RIFF 头后的 WEBP 标记
            if mime == "webp":
                if len(data) >= 12 and data[8:12] == b"WEBP":
                    return "webp"
                continue
            return mime
    return "png"  # 默认 PNG


def _image_to_base64(data: bytes) -> str:
    """将图片二进制数据转为 ``data:image/{format};base64,...`` 格式字符串。

    尽量保留原始格式（JPEG 不会被重编码为 PNG），以避免体积膨胀。

    Args:
        data: 图片二进制数据。

    Returns:
        ``data:image/{mime};base64,...`` 格式的完整 data URL。
    """
    mime = _detect_image_mime(data)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{b64}"
