from __future__ import annotations
import hashlib
import json
from typing import Any, Dict

def hash_tables_meta(meta: Dict[str, Any]) -> str:
    s = json.dumps(meta, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
