import os
from langchain_core.embeddings import Embeddings


def get_embeddings() -> Embeddings:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=api_key,
        )
    else:
        from langchain_ollama import OllamaEmbeddings
        from config import OLLAMA_BASE_URL, EMBEDDING_MODEL
        return OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=EMBEDDING_MODEL,
        )

