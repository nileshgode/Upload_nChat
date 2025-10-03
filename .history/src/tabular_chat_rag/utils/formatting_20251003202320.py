from __future__ import annotations
import pandas as pd

def to_markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if len(df) > max_rows:
        df = df.head(max_rows)
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback when tabulate extras are missing
        return df.head(max_rows).to_string(index=False)
