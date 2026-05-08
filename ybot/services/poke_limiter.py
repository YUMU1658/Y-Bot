"""戳一戳速率限制器。

提供 per-target 冷却和每日总次数上限，防止戳一戳滥用。
纯内存实现，重启后重置（与 QQ 官方行为一致）。
"""

from __future__ import annotations

import time
from datetime import datetime


class PokeLimiter:
    """戳一戳速率限制器。

    - 对同一 target_id 的冷却时间（per-user cooldown）
    - 每日总次数上限（daily limit）
    """

    def __init__(self, cooldown: int = 10, daily_limit: int = 200) -> None:
        self._cooldown = cooldown
        self._daily_limit = daily_limit
        self._last_poke: dict[int, float] = {}  # target_id → 上次戳的时间戳
        self._daily_count: int = 0
        self._daily_reset_date: str = ""  # "YYYY-MM-DD" 格式

    def check(self, target_id: int) -> str | None:
        """检查是否可以戳指定用户。

        Returns:
            None 表示可以戳；否则返回失败原因字符串。
        """
        now = time.time()
        today = datetime.now().strftime("%Y-%m-%d")

        # 每日重置
        if today != self._daily_reset_date:
            self._daily_count = 0
            self._daily_reset_date = today
            # 顺便清理过期的冷却记录，防止 _last_poke 无界增长
            self._last_poke = {
                tid: ts for tid, ts in self._last_poke.items()
                if now - ts < self._cooldown
            }

        # 每日上限
        if self._daily_count >= self._daily_limit:
            return f"已达到每日戳一戳上限({self._daily_limit}次)"

        # 同用户冷却
        last = self._last_poke.get(target_id)
        if last is not None:
            elapsed = now - last
            if elapsed < self._cooldown:
                remaining = self._cooldown - elapsed
                return f"戳一戳冷却中，请等待{remaining:.0f}秒"

        return None

    def record(self, target_id: int) -> None:
        """记录一次成功的戳一戳。"""
        self._last_poke[target_id] = time.time()
        self._daily_count += 1
