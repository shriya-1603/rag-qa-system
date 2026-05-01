import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from app.embeddings import get_embeddings
from config import VECTORSTORE_DIR


def _index_path() -> str:
    return VECTORSTORE_DIR


def build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(_index_path())
    return vs


def load_vectorstore() -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(_index_path(), embeddings, allow_dangerous_deserialization=True)


def add_to_vectorstore(chunks: List[Document]) -> FAISS:
    if os.path.exists(os.path.join(_index_path(), "index.faiss")):
        vs = load_vectorstore()
        vs.add_documents(chunks)
    else:
        vs = build_vectorstore(chunks)
    vs.save_local(_index_path())
    return vs


def vectorstore_exists() -> bool:
    return os.path.exists(os.path.join(_index_path(), "index.faiss"))
