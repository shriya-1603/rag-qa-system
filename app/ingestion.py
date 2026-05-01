from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_document(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def ingest_file(file_path: str) -> List[Document]:
    docs = load_document(file_path)
    return chunk_documents(docs)
