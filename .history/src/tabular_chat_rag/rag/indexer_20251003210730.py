from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from ..config import SETTINGS

def _chunk_rows(df: pd.DataFrame, chunk_rows: int) -> List[Tuple[int, int, str]]:
    chunks = []
    for start in range(0, len(df), chunk_rows):
        end = min(start + chunk_rows, len(df))
        text = df.iloc[start:end].to_csv(index=False)
        chunks.append((start, end, text))
    return chunks

class RowChunkIndex:
    def __init__(self, embed_model: str = SETTINGS.embed_model):
        self.model = SentenceTransformer(embed_model)
        self.index = None
        self.texts: List[str] = []
        self.meta: List[dict] = []

    def build(self, tables: Dict[str, pd.DataFrame], chunk_rows: int = SETTINGS.chunk_rows):
        embeddings = []
        self.texts.clear()
        self.meta.clear()
        for table, df in tables.items():
            for start, end, text in _chunk_rows(df, chunk_rows):
                self.texts.append(text)
                self.meta.append({"table": table, "start": start, "end": end})
                emb = self.model.encode(text, normalize_embeddings=True)
                embeddings.append(emb.astype("float32"))
        if not embeddings:
            self.index = None
            return
        mat = np.vstack(embeddings)
        self.index = faiss.IndexFlatIP(mat.shape[1])
        self.index.add(mat)

    def search(self, query: str, k: int) -> List[dict]:
        if self.index is None:
            return []
        q = self.model.encode(query, normalize_embeddings=True).astype("float32")[None, :]
        sims, idxs = self.index.search(q, k)
        results = []
        for i, score in zip(idxs[0], sims[0]):
            if i == -1:
                continue
            results.append({**self.meta[i], "text": self.texts[i], "score": float(score)})
        return results
