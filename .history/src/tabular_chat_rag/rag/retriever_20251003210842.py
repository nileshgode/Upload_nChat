from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate
from ..llm.ollama_client import get_llm
from ..config import SETTINGS
from .indexer import RowChunkIndex

SYSTEM_PROMPT = """You are a data assistant. Use ONLY the provided context chunks to answer.
If numeric accuracy is required and context is insufficient, recommend switching to the SQL path.
Be concise and include a brief rationale at the end.
"""

def answer_with_rag(question: str, idx: RowChunkIndex, k: int = SETTINGS.retrieval_k, model: str = SETTINGS.ollama_model) -> str:
    chunks = idx.search(question, k=k)
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    llm = get_llm(model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", f"Context chunks:\n{context}\n\nQuestion: {question}"),
    ])
    return (prompt | llm).invoke({}).content
