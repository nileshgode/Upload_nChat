from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from ..llm.ollama_client import get_llm
from ..config import SETTINGS

SYSTEM_PROMPT = """Write minimal, correct pandas code to answer using given DataFrames.
- DataFrames are available using variables named after tables.
- Assign final answer to one of: result_df, result_value.
- No imports, no IO, no network/system calls.
- Return ONLY Python code between <code> and </code>.
"""

EXAMPLE = """
<code>
# Example:
# result_value = orders['amount'].sum()
</code>
""".strip()

def _schema_comment(tables: Dict[str, pd.DataFrame]) -> str:
    lines = []
    for name, df in tables.items():
        cols = ", ".join(f"{c}:{str(t)}" for c, t in df.dtypes.items())
        lines.append(f"# {name} -> {cols} (rows={len(df)})")
    return "\n".join(lines)

def plan_code(question: str, tables: Dict[str, pd.DataFrame], model: str = SETTINGS.ollama_model) -> str:
    llm = get_llm(model)
    schema = _schema_comment(tables)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", f"{schema}\n\nQuestion: {question}\n\n{EXAMPLE}\nReturn code in <code>...</code>"),
    ])
    out = (prompt | llm).invoke({})
    text = out.content
    s, e = text.find("<code>"), text.find("</code>")
    code = text[s+6:e].strip() if s != -1 and e != -1 else text.strip()
    return code

def run_code(code: str, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    # Restricted exec environment
    safe_globals = {"__builtins__": {"len": len, "min": min, "max": max, "sum": sum, "range": range}}
    safe_locals = {**tables}
    exec(code, safe_globals, safe_locals)
    out: Dict[str, Any] = {}
    if "result_df" in safe_locals and isinstance(safe_locals["result_df"], pd.DataFrame):
        out["result_df"] = safe_locals["result_df"]
    if "result_value" in safe_locals and not isinstance(safe_locals["result_value"], pd.DataFrame):
        out["result_value"] = safe_locals["result_value"]
    return out
