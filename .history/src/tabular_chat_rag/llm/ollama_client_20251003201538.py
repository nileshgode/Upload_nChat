from typing import Optional
from langchain_ollama import ChatOllama

def get_llm(model_name: str, temperature: float = 0.2, timeout: Optional[float] = 90.0):
    # Phi-3 via Ollama; ensure `ollama pull phi3` and `ollama serve` are running
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        request_timeout=timeout,
    )
