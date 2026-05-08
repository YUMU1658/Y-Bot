"""世界书 (World Book / Lorebook) 服务模块。

动态知识注入系统，根据对话上下文自动匹配关键词/正则/常驻条目，
将补充信息注入到 prompt 的指定位置。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ybot.constants import VALID_ROLES
from ybot.utils.logger import get_logger

logger = get_logger("WorldBook")


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class TriggerConfig:
    """触发配置。"""

    mode: str = "keyword"  # keyword / regex / constant / disabled
    keywords: list[str] = field(default_factory=list)
    secondary_keywords: list[str] = field(default_factory=list)
    regex: str | None = None
    filter_in: str | None = None
    filter_out: str | None = None


@dataclass
class InsertionConfig:
    """插入配置。"""

    position: str = "system_after"
    depth: int = 0
    role: str = "system"
    order: int = 500


@dataclass
class EntryOptions:
    """条目高级选项。"""

    constant: bool = False
    disabled: bool = False
    selective: bool = False
    exclude_recursion: bool = False
    prevent_recursion: bool = False
    delay_until_recursion: bool = False
    probability: int = 100
    cooldown: int = 0
    group: str | None = None
    group_weight: int = 100
    sticky: int = 0
    use_group_scoring: bool = False


@dataclass
class WorldBookSettings:
    """世界书全局扫描设置。"""

    scan_depth: int = 2
    token_budget: int = 2048
    recursive_scanning: bool = True
    max_recursion_depth: int = 3
    case_sensitive: bool = False
    match_whole_words: bool = False


@dataclass
class WorldBookEntry:
    """世界书条目。"""

    id: str
    name: str
    comment: str
    enabled: bool
    content: str
    trigger: TriggerConfig
    insertion: InsertionConfig
    options: EntryOptions
    # 运行时缓存（不序列化）
    _source_book: str = ""
    _compiled_regex: re.Pattern[str] | None = field(
        default=None, repr=False, compare=False
    )
    _compiled_filter_in: re.Pattern[str] | None = field(
        default=None, repr=False, compare=False
    )
    _compiled_filter_out: re.Pattern[str] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass
class WorldBook:
    """一本世界书。"""

    schema_version: int
    id: str
    name: str
    description: str
    enabled: bool
    settings: WorldBookSettings
    entries: list[WorldBookEntry]


@dataclass
class ActivatedEntry:
    """已激活的世界书条目（准备注入）。"""

    entry: WorldBookEntry
    matched_keywords: list[str] = field(default_factory=list)
    recursion_depth: int = 0


@dataclass
class SessionWorldBookState:
    """每个会话的世界书运行时状态。"""

    cooldowns: dict[str, int] = field(default_factory=dict)  # entry_id → 剩余冷却轮数
    sticky: dict[str, int] = field(default_factory=dict)  # entry_id → 剩余粘性轮数
    last_activated: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# 世界书服务
# ---------------------------------------------------------------------------

# 世界书条目可用的插入位置
_VALID_WB_POSITIONS = {
    "system_before",
    "system_after",
    "system_end",
    "messages_start",
    "messages_end",
    "at_depth",
    "user_before",
    "user_after",
    "assistant_before",
    "assistant_after",
}


class WorldBookService:
    """世界书服务：加载世界书、扫描上下文、收集激活条目。"""

    def __init__(
        self,
        worldbook_dir: str = "config/worldbooks",
        enabled_books: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self._worldbook_dir = Path(worldbook_dir)
        self._enabled_books = enabled_books or []
        self._enabled = enabled
        self._books: list[WorldBook] = []
        self._session_states: dict[str, SessionWorldBookState] = {}

    # ---- 公共接口 ----

    def is_enabled(self) -> bool:
        """是否启用世界书系统。"""
        return self._enabled

    def load(self) -> None:
        """加载所有启用的世界书文件。"""
        self._books.clear()
        self._ensure_worldbook_dir()

        if not self._worldbook_dir.exists():
            return

        json_files = sorted(self._worldbook_dir.glob("*.json"))
        if not json_files:
            logger.info(f"世界书目录 {self._worldbook_dir} 中没有 JSON 文件")
            return

        for path in json_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                book = self._parse_worldbook(data, path.name)
                if not book.enabled:
                    logger.debug(f"跳过已禁用的世界书: {book.name} ({path.name})")
                    continue
                if self._enabled_books and book.id not in self._enabled_books:
                    logger.debug(
                        f"跳过未在 enabled_books 中的世界书: {book.name} ({path.name})"
                    )
                    continue
                self._books.append(book)
                entry_count = sum(1 for e in book.entries if e.enabled)
                logger.info(
                    f"已加载世界书: {book.name} ({entry_count} 条有效条目)"
                )
            except Exception as e:
                logger.error(f"加载世界书文件失败 ({path.name}): {e}")

        total = sum(len(b.entries) for b in self._books)
        logger.info(f"世界书加载完成: {len(self._books)} 本, 共 {total} 条条目")

    def reload(self) -> None:
        """热重载世界书。"""
        logger.info("正在重载世界书...")
        self.load()

    def scan_and_collect(
        self,
        *,
        current_message: str,
        history: list[dict[str, Any]],
        session_key: str,
    ) -> list[ActivatedEntry]:
        """扫描上下文，收集所有触发的世界书条目。

        Args:
            current_message: 当前用户消息文本。
            history: 对话历史消息列表。
            session_key: 会话标识（用于冷却/粘性状态）。

        Returns:
            按 (position, order, entry_id) 排序的已激活条目列表。
        """
        if not self._books:
            return []

        state = self._session_states.setdefault(
            session_key, SessionWorldBookState()
        )

        all_activated: list[ActivatedEntry] = []

        for book in self._books:
            activated = self._scan_book(book, current_message, history, state)
            all_activated.extend(activated)

        # 按 (position, order, entry_id) 排序
        all_activated.sort(
            key=lambda a: (a.entry.insertion.position, a.entry.insertion.order, a.entry.id)
        )

        # 更新运行时状态
        self._update_state(state, all_activated)

        return all_activated

    # ---- 内部方法 ----

    def _scan_book(
        self,
        book: WorldBook,
        current_message: str,
        history: list[dict[str, Any]],
        state: SessionWorldBookState,
    ) -> list[ActivatedEntry]:
        """扫描单本世界书。"""
        settings = book.settings

        # 1. 构建扫描文本
        scan_text = self._build_scan_text(current_message, history, settings.scan_depth)

        # 准备大小写处理
        if not settings.case_sensitive:
            scan_text_lower = scan_text.lower()
        else:
            scan_text_lower = scan_text

        # 2. 第一轮扫描（直接匹配）
        activated: list[ActivatedEntry] = []
        activated_ids: set[str] = set()
        remaining_entries: list[WorldBookEntry] = []

        for entry in book.entries:
            if not entry.enabled:
                continue
            if entry.options.disabled or entry.trigger.mode == "disabled":
                continue

            # 常驻条目
            if entry.options.constant or entry.trigger.mode == "constant":
                activated.append(ActivatedEntry(entry=entry, recursion_depth=0))
                activated_ids.add(entry.id)
                continue

            # 延迟到递归阶段的条目
            if entry.options.delay_until_recursion:
                remaining_entries.append(entry)
                continue

            # 普通匹配
            matched_kws = self._match_entry(entry, scan_text, scan_text_lower, settings)
            if matched_kws is not None:
                # 过滤器检查
                if not self._apply_filters(entry, scan_text):
                    continue
                activated.append(
                    ActivatedEntry(
                        entry=entry, matched_keywords=matched_kws, recursion_depth=0
                    )
                )
                activated_ids.add(entry.id)
            else:
                remaining_entries.append(entry)

        # 3. 递归扫描
        if settings.recursive_scanning and remaining_entries:
            self._recursive_scan(
                remaining_entries,
                activated,
                activated_ids,
                settings,
                scan_text,
            )

        # 4. 互斥组解析
        activated = self._resolve_groups(activated)

        # 5. 概率过滤
        activated = self._apply_probability(activated)

        # 6. 冷却检查
        activated = self._apply_cooldown(activated, state)

        # 7. 粘性检查
        activated = self._apply_sticky(activated, state, book)

        # 8. Token 预算裁剪
        activated = self._enforce_budget(activated, settings.token_budget)

        return activated

    def _build_scan_text(
        self,
        current_message: str,
        history: list[dict[str, Any]],
        scan_depth: int,
    ) -> str:
        """构建扫描文本：当前消息 + 最近 N 条历史。"""
        parts = [current_message]

        if scan_depth > 0 and history:
            # 从历史末尾取最近 scan_depth 条（不含最后一条，因为那是当前消息）
            # history 中最后一条是当前用户消息（已在 current_message 中），
            # 所以从 history[:-1] 的末尾取
            relevant = history[:-1] if len(history) > 1 else []
            for msg in relevant[-scan_depth:]:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # multimodal content: 提取文本部分
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))

        return "\n".join(parts)

    def _match_entry(
        self,
        entry: WorldBookEntry,
        scan_text: str,
        scan_text_lower: str,
        settings: WorldBookSettings,
    ) -> list[str] | None:
        """匹配单个条目。返回匹配到的关键词列表，未匹配返回 None。"""
        mode = entry.trigger.mode

        if mode == "keyword":
            return self._match_keywords(entry, scan_text, scan_text_lower, settings)
        elif mode == "regex":
            return self._match_regex(entry, scan_text)

        return None

    def _match_keywords(
        self,
        entry: WorldBookEntry,
        scan_text: str,
        scan_text_lower: str,
        settings: WorldBookSettings,
    ) -> list[str] | None:
        """关键词匹配。"""
        if not entry.trigger.keywords:
            return None

        matched: list[str] = []
        for kw in entry.trigger.keywords:
            check_kw = kw if settings.case_sensitive else kw.lower()
            check_text = scan_text if settings.case_sensitive else scan_text_lower

            if settings.match_whole_words:
                # 全词匹配（仅对英文有效）
                pattern = rf"\b{re.escape(check_kw)}\b"
                if re.search(pattern, check_text):
                    matched.append(kw)
            else:
                if check_kw in check_text:
                    matched.append(kw)

        if not matched:
            return None

        # 选择性触发：还需次要关键词匹配
        if entry.options.selective and entry.trigger.secondary_keywords:
            secondary_matched = False
            for kw in entry.trigger.secondary_keywords:
                check_kw = kw if settings.case_sensitive else kw.lower()
                check_text = scan_text if settings.case_sensitive else scan_text_lower

                if settings.match_whole_words:
                    pattern = rf"\b{re.escape(check_kw)}\b"
                    if re.search(pattern, check_text):
                        secondary_matched = True
                        break
                else:
                    if check_kw in check_text:
                        secondary_matched = True
                        break

            if not secondary_matched:
                return None

        return matched

    def _match_regex(
        self,
        entry: WorldBookEntry,
        scan_text: str,
    ) -> list[str] | None:
        """正则匹配。"""
        if not entry.trigger.regex:
            return None

        try:
            compiled = entry._compiled_regex
            if compiled is None:
                return None
            match = compiled.search(scan_text)
            if match:
                return [match.group()]
        except re.error:
            logger.warning(f"世界书条目 {entry.id} 的正则表达式无效: {entry.trigger.regex}")

        return None

    def _apply_filters(self, entry: WorldBookEntry, scan_text: str) -> bool:
        """应用 filter_in / filter_out 过滤器。返回 True 表示通过。"""
        # filter_in: 必须匹配才有资格
        if entry._compiled_filter_in is not None:
            try:
                if not entry._compiled_filter_in.search(scan_text):
                    return False
            except re.error:
                pass

        # filter_out: 匹配则排除
        if entry._compiled_filter_out is not None:
            try:
                if entry._compiled_filter_out.search(scan_text):
                    return False
            except re.error:
                pass

        return True

    def _recursive_scan(
        self,
        remaining_entries: list[WorldBookEntry],
        activated: list[ActivatedEntry],
        activated_ids: set[str],
        settings: WorldBookSettings,
        original_scan_text: str,
    ) -> None:
        """递归扫描：已激活条目的内容也参与关键词匹配。"""
        for depth in range(1, settings.max_recursion_depth + 1):
            # 构建递归扫描文本：所有已激活条目的内容（排除 exclude_recursion 的）
            recursive_parts = []
            for act in activated:
                if not act.entry.options.exclude_recursion:
                    recursive_parts.append(act.entry.content)

            if not recursive_parts:
                break

            recursive_text = "\n".join(recursive_parts)
            if not settings.case_sensitive:
                recursive_text_lower = recursive_text.lower()
            else:
                recursive_text_lower = recursive_text

            new_matches: list[ActivatedEntry] = []
            still_remaining: list[WorldBookEntry] = []

            for entry in remaining_entries:
                if entry.id in activated_ids:
                    continue

                # prevent_recursion 的条目不能被递归触发
                if entry.options.prevent_recursion:
                    still_remaining.append(entry)
                    continue

                matched_kws = self._match_entry(
                    entry, recursive_text, recursive_text_lower, settings
                )
                if matched_kws is not None:
                    if self._apply_filters(entry, original_scan_text):
                        act = ActivatedEntry(
                            entry=entry,
                            matched_keywords=matched_kws,
                            recursion_depth=depth,
                        )
                        new_matches.append(act)
                        activated_ids.add(entry.id)
                    else:
                        still_remaining.append(entry)
                else:
                    still_remaining.append(entry)

            if not new_matches:
                break

            activated.extend(new_matches)
            remaining_entries = still_remaining

    def _resolve_groups(
        self, activated: list[ActivatedEntry]
    ) -> list[ActivatedEntry]:
        """互斥组解析：同组内只保留一个条目。"""
        groups: dict[str, list[ActivatedEntry]] = {}
        ungrouped: list[ActivatedEntry] = []

        for act in activated:
            group = act.entry.options.group
            if group:
                groups.setdefault(group, []).append(act)
            else:
                ungrouped.append(act)

        result = list(ungrouped)

        for group_name, members in groups.items():
            if len(members) == 1:
                result.append(members[0])
                continue

            # 检查是否使用评分模式
            use_scoring = any(m.entry.options.use_group_scoring for m in members)

            if use_scoring:
                # 评分模式：选择权重最高的
                best = max(members, key=lambda m: m.entry.options.group_weight)
                result.append(best)
            else:
                # 加权随机选择
                weights = [m.entry.options.group_weight for m in members]
                total = sum(weights)
                if total <= 0:
                    result.append(random.choice(members))
                else:
                    chosen = random.choices(members, weights=weights, k=1)[0]
                    result.append(chosen)

        return result

    def _apply_probability(
        self, activated: list[ActivatedEntry]
    ) -> list[ActivatedEntry]:
        """概率过滤。"""
        result: list[ActivatedEntry] = []
        for act in activated:
            prob = act.entry.options.probability
            if prob >= 100:
                result.append(act)
            elif prob <= 0:
                continue
            elif random.randint(1, 100) <= prob:
                result.append(act)
        return result

    def _apply_cooldown(
        self,
        activated: list[ActivatedEntry],
        state: SessionWorldBookState,
    ) -> list[ActivatedEntry]:
        """冷却检查：移除仍在冷却中的条目。"""
        result: list[ActivatedEntry] = []
        for act in activated:
            remaining = state.cooldowns.get(act.entry.id, 0)
            if remaining > 0:
                # 仍在冷却中，跳过（除非粘性仍有效）
                sticky_remaining = state.sticky.get(act.entry.id, 0)
                if sticky_remaining > 0:
                    result.append(act)
                # 否则跳过
            else:
                result.append(act)
        return result

    def _apply_sticky(
        self,
        activated: list[ActivatedEntry],
        state: SessionWorldBookState,
        book: WorldBook,
    ) -> list[ActivatedEntry]:
        """粘性检查：将仍在粘性期内的条目重新加入。"""
        activated_ids = {act.entry.id for act in activated}
        result = list(activated)

        # 检查粘性状态中的条目
        for entry_id, remaining in list(state.sticky.items()):
            if remaining > 0 and entry_id not in activated_ids:
                # 查找对应的条目
                entry = self._find_entry(book, entry_id)
                if entry and entry.enabled:
                    result.append(
                        ActivatedEntry(entry=entry, recursion_depth=0)
                    )

        return result

    def _enforce_budget(
        self, activated: list[ActivatedEntry], token_budget: int
    ) -> list[ActivatedEntry]:
        """Token 预算裁剪：按 order 排序，累加内容长度，超出预算时截断。

        使用 len(content) 作为粗略的 token 估算（中文约 1 字符 ≈ 1-2 token）。
        """
        if token_budget <= 0:
            return activated

        # 按 order 排序（小的优先保留）
        sorted_entries = sorted(activated, key=lambda a: a.entry.insertion.order)

        result: list[ActivatedEntry] = []
        used = 0

        for act in sorted_entries:
            cost = len(act.entry.content)
            if used + cost <= token_budget:
                result.append(act)
                used += cost
            else:
                logger.debug(
                    f"世界书条目 {act.entry.id} 因 token 预算不足被裁剪 "
                    f"(已用 {used}, 需要 {cost}, 预算 {token_budget})"
                )

        return result

    def _update_state(
        self,
        state: SessionWorldBookState,
        activated: list[ActivatedEntry],
    ) -> None:
        """更新会话的运行时状态（冷却/粘性计数器）。"""
        current_ids = {act.entry.id for act in activated}

        # 递减所有冷却计数器
        for entry_id in list(state.cooldowns.keys()):
            state.cooldowns[entry_id] -= 1
            if state.cooldowns[entry_id] <= 0:
                del state.cooldowns[entry_id]

        # 递减所有粘性计数器
        for entry_id in list(state.sticky.keys()):
            state.sticky[entry_id] -= 1
            if state.sticky[entry_id] <= 0:
                del state.sticky[entry_id]

        # 为新激活的条目设置冷却和粘性
        for act in activated:
            entry = act.entry
            if entry.options.cooldown > 0:
                state.cooldowns[entry.id] = entry.options.cooldown

            if entry.options.sticky > 0 and entry.id not in state.sticky:
                state.sticky[entry.id] = entry.options.sticky

        state.last_activated = current_ids

    def _find_entry(self, book: WorldBook, entry_id: str) -> WorldBookEntry | None:
        """在世界书中查找指定 ID 的条目。"""
        for entry in book.entries:
            if entry.id == entry_id:
                return entry
        return None

    # ---- 解析逻辑 ----

    def _parse_worldbook(self, data: dict[str, Any], filename: str) -> WorldBook:
        """从 JSON 字典解析世界书。"""
        settings_data = data.get("settings", {})
        settings = WorldBookSettings(
            scan_depth=int(settings_data.get("scan_depth", 2)),
            token_budget=int(settings_data.get("token_budget", 2048)),
            recursive_scanning=bool(settings_data.get("recursive_scanning", True)),
            max_recursion_depth=int(settings_data.get("max_recursion_depth", 3)),
            case_sensitive=bool(settings_data.get("case_sensitive", False)),
            match_whole_words=bool(settings_data.get("match_whole_words", False)),
        )

        entries: list[WorldBookEntry] = []
        book_id = str(data.get("id", filename))

        for raw_entry in data.get("entries", []):
            if not isinstance(raw_entry, dict):
                logger.warning(f"跳过非法世界书条目 (in {filename}): entry 不是对象")
                continue
            try:
                entry = self._parse_entry(raw_entry, book_id, settings)
                entries.append(entry)
            except Exception as e:
                logger.warning(
                    f"跳过非法世界书条目 (in {filename}): {e}"
                )

        return WorldBook(
            schema_version=int(data.get("schema_version", 1)),
            id=book_id,
            name=str(data.get("name", filename)),
            description=str(data.get("description", "")),
            enabled=bool(data.get("enabled", True)),
            settings=settings,
            entries=entries,
        )

    def _parse_entry(
        self,
        raw: dict[str, Any],
        book_id: str,
        settings: WorldBookSettings,
    ) -> WorldBookEntry:
        """解析单个世界书条目。"""
        trigger_data = raw.get("trigger", {})
        insertion_data = raw.get("insertion", {})
        options_data = raw.get("options", {})

        trigger = TriggerConfig(
            mode=str(trigger_data.get("mode", "keyword")),
            keywords=list(trigger_data.get("keywords", [])),
            secondary_keywords=list(trigger_data.get("secondary_keywords", [])),
            regex=trigger_data.get("regex"),
            filter_in=trigger_data.get("filter_in"),
            filter_out=trigger_data.get("filter_out"),
        )

        position = str(insertion_data.get("position", "system_after"))
        if position not in _VALID_WB_POSITIONS:
            logger.warning(
                f"世界书条目 {raw.get('id', '?')} 使用了未知位置 '{position}'，"
                f"回退到 'system_after'"
            )
            position = "system_after"

        role = str(insertion_data.get("role", "system"))
        if role not in VALID_ROLES:
            logger.warning(
                f"世界书条目 {raw.get('id', '?')} 使用了未知角色 '{role}'，"
                f"回退到 'system'"
            )
            role = "system"

        insertion = InsertionConfig(
            position=position,
            depth=int(insertion_data.get("depth", 0)),
            role=role,
            order=int(insertion_data.get("order", 500)),
        )

        options = EntryOptions(
            constant=bool(options_data.get("constant", False)),
            disabled=bool(options_data.get("disabled", False)),
            selective=bool(options_data.get("selective", False)),
            exclude_recursion=bool(options_data.get("exclude_recursion", False)),
            prevent_recursion=bool(options_data.get("prevent_recursion", False)),
            delay_until_recursion=bool(
                options_data.get("delay_until_recursion", False)
            ),
            probability=int(options_data.get("probability", 100)),
            cooldown=int(options_data.get("cooldown", 0)),
            group=options_data.get("group"),
            group_weight=int(options_data.get("group_weight", 100)),
            sticky=int(options_data.get("sticky", 0)),
            use_group_scoring=bool(options_data.get("use_group_scoring", False)),
        )

        entry = WorldBookEntry(
            id=str(raw.get("id", "")),
            name=str(raw.get("name", "")),
            comment=str(raw.get("comment", "")),
            enabled=bool(raw.get("enabled", True)),
            content=str(raw.get("content", "")),
            trigger=trigger,
            insertion=insertion,
            options=options,
            _source_book=book_id,
        )

        # 预编译正则表达式
        re_flags = 0 if settings.case_sensitive else re.IGNORECASE
        if trigger.regex:
            try:
                entry._compiled_regex = re.compile(trigger.regex, re_flags)
            except re.error as e:
                logger.warning(
                    f"世界书条目 {entry.id} 的正则表达式编译失败: {e}"
                )
        if trigger.filter_in:
            try:
                entry._compiled_filter_in = re.compile(trigger.filter_in, re_flags)
            except re.error as e:
                logger.warning(
                    f"世界书条目 {entry.id} 的 filter_in 正则编译失败: {e}"
                )
        if trigger.filter_out:
            try:
                entry._compiled_filter_out = re.compile(trigger.filter_out, re_flags)
            except re.error as e:
                logger.warning(
                    f"世界书条目 {entry.id} 的 filter_out 正则编译失败: {e}"
                )

        return entry

    def _ensure_worldbook_dir(self) -> None:
        """确保世界书目录存在。"""
        if not self._worldbook_dir.exists():
            self._worldbook_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"已创建世界书目录: {self._worldbook_dir}")
