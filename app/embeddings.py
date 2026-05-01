from langchain_ollama import OllamaEmbeddings
from config import OLLAMA_BASE_URL, EMBEDDING_MODEL


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=EMBEDDING_MODEL,
    )
