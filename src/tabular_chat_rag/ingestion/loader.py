from __future__ import annotations
from typing import Dict, Tuple, Any
import io
import pandas as pd

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df

def load_file(uploaded) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    name = uploaded.name.lower()
    tables: Dict[str, pd.DataFrame] = {}
    meta: Dict[str, Any] = {"sources": []}

    if name.endswith((".xlsx", ".xls")):
        xls = pd.read_excel(uploaded, sheet_name=None, engine="openpyxl")
        for sheet, df in xls.items():
            df = _normalize_columns(df)
            tables[sheet] = df
            meta["sources"].append({"table": sheet, "rows": len(df), "cols": list(df.columns)})
    elif name.endswith(".csv"):
        buf = io.BytesIO(uploaded.read())
        df = pd.read_csv(buf, dtype_backend="pyarrow")
        df = _normalize_columns(df)
        base = name.rsplit(".", 1)[0]
        tables[base] = df
        meta["sources"].append({"table": base, "rows": len(df), "cols": list(df.columns)})
    else:
        raise ValueError("Unsupported file type. Please upload .csv or .xlsx")

    return tables, meta
