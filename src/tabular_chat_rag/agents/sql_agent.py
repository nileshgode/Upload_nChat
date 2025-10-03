from __future__ import annotations
from typing import Dict, Tuple
import duckdb
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from ..llm.ollama_client import get_llm
from ..utils.safety import sql_is_select_only, enforce_limit
from ..config import SETTINGS

SYSTEM_PROMPT = """You write correct DuckDB SQL over provided tables.
Rules:
- Use only SELECT queries.
- Prefer explicit column names and qualified references for joins.
- Add LIMIT for raw row outputs unless aggregation is requested.
- Return ONLY SQL between <sql> and </sql>.
"""

def _build_schema_text(tables: Dict[str, pd.DataFrame]) -> str:
    parts = []
    for name, df in tables.items():
        cols = ", ".join(f"{c} {str(t)}" for c, t in df.dtypes.items())
        parts.append(f"{name}({cols}) rows={len(df)}")
    return "\n".join(parts)

def plan_sql(question: str, tables: Dict[str, pd.DataFrame], model: str = SETTINGS.ollama_model) -> str:
    llm = get_llm(model)
    schema = _build_schema_text(tables)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", f"Schema:\n{schema}\n\nQuestion: {question}\n\nReturn SQL in <sql>...</sql>"),
    ])
    out = (prompt | llm).invoke({})
    text = out.content
    s, e = text.find("<sql>"), text.find("</sql>")
    sql = text[s+5:e].strip() if s != -1 and e != -1 else text.strip()
    sql = enforce_limit(sql, SETTINGS.sql_default_limit)
    if not sql_is_select_only(sql):
        raise ValueError("Generated SQL is not SELECT-only. Please rephrase.")
    return sql

def run_sql(sql: str, tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, duckdb.DuckDBPyConnection]:
    con = duckdb.connect(":memory:")
    for name, df in tables.items():
        con.register(name, df)
    result = con.execute(sql).df()
    return result, con
