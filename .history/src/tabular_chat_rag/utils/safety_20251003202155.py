from __future__ import annotations

FORBIDDEN_SQL_TOKENS = [
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "copy", "truncate", "replace", "merge", "vacuum"
]

def sql_is_select_only(sql: str) -> bool:
    s = " ".join(sql.strip().lower().split())
    if not s.startswith("select"):
        return False
    return not any(tok in s for tok in FORBIDDEN_SQL_TOKENS)

def enforce_limit(sql: str, default_limit: int = 200) -> str:
    s = sql.strip().rstrip(";")
    low = s.lower()
    if " limit " in low or "count(" in low or "sum(" in low or "avg(" in low or "min(" in low or "max(" in low:
        return s
    return f"{s} LIMIT {default_limit}"
