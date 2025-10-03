from __future__ import annotations
from typing import Dict, List
import pandas as pd

def head_preview(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return df.head(n)

def schema_line(name: str, df: pd.DataFrame) -> str:
    dtypes = ", ".join(f"{c}:{str(t)}" for c, t in df.dtypes.items())
    return f"table {name}(cols: {dtypes}; rows: {len(df)})"

def schema_summary(tables: Dict[str, pd.DataFrame]) -> List[str]:
    return [schema_line(name, df) for name, df in tables.items()]
