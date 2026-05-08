"""共享常量模块。

集中定义跨模块使用的常量，消除重复定义。
"""

from __future__ import annotations

from datetime import timedelta, timezone

# 北京时间 UTC+8（统一时区常量，替代各模块独立定义的 _CST / _TZ_CST）
TZ_CST = timezone(timedelta(hours=8))

# 合法的消息角色
VALID_ROLES = frozenset({"system", "user", "assistant"})

# 预设/世界书的合法插入位置
VALID_POSITIONS = frozenset({
    "system_before",
    "system_after",
    "user_before",
    "user_after",
    "assistant_before",
    "assistant_after",
    "messages_start",
    "messages_end",
})

# 群角色中文映射
ROLE_LABEL_MAP: dict[str, str] = {
    "owner": "群主",
    "admin": "管理员",
    "member": "成员",
}

# 工具层分页默认值
DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200

# 性别字段中文映射
SEX_LABEL_MAP: dict[str, str] = {
    "male": "男",
    "female": "女",
}
