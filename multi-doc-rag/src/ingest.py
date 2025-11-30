from typing import List
from pathlib import Path
from pydantic import BaseModel
import logging

# LangChain Loaders
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,  
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document

# Docling Imports
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker

from .utils import create_text_splitter

logger = logging.getLogger(__name__)


class IngestConfig(BaseModel):
    """Configuration for document chunking."""
    max_chunk_size: int = 1000
    chunk_overlap: int = 200


def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Load multiple file formats into LangChain Document objects.
    Uses Docling for structured formats (PDF, DOCX, PPTX, HTML)
    and standard LangChain loaders for others.
    """
    docs = []

    for path in file_paths:
        ext = Path(path).suffix.lower()
        try:
            # --- Prefer Docling for structured formats ---
            if ext in [".pdf", ".docx", ".pptx", ".html"]:
                loader = DoclingLoader(
                    file_path=path,
                    export_type=ExportType.DOC_CHUNKS,  # chunked export for embeddings
                    chunker=HybridChunker(),             # uses AI-based hybrid chunking
                )

            # --- Simple loaders for plain formats ---
            elif ext in [".txt", ".md"]:
                loader = TextLoader(path, encoding="utf-8")

            elif ext == ".csv":
                loader = CSVLoader(file_path=path)

            elif ext in [".xls", ".xlsx"]:
                loader = UnstructuredExcelLoader(path, mode="single")

            else:
                logger.warning(f"Unsupported file extension {ext}. Using TextLoader fallback.")
                loader = TextLoader(path, encoding="utf-8")

            # Load documents
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            logger.info(f"Loaded {len(loaded_docs)} document(s) from {path}")

        except Exception as e:
            logger.exception(f"Failed to load {path}: {e}")
            continue

    return docs


def chunk_documents(docs: List[Document], cfg: IngestConfig) -> List[Document]:
    """
    Split loaded documents into smaller, overlapping chunks for embedding.
    Docling already provides AI-based chunking for structured docs,
    so this is mainly used for plain text formats.
    """
    splitter = create_text_splitter(cfg.max_chunk_size, cfg.chunk_overlap)
    split_docs = []

    for d in docs:
        chunks = splitter.split_documents([d])
        split_docs.extend(chunks)

    return split_docs
