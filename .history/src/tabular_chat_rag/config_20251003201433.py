from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    default_pipeline: str = "sql"        # "df" | "sql" | "rag"
    ollama_model: str = "phi3"           # Using Ollama phi-3 family
    max_rows_df_path: int = 200_000
    chunk_rows: int = 500
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_k: int = 4
    max_preview_rows: int = 20
    sql_default_limit: int = 200

SETTINGS = Settings()
