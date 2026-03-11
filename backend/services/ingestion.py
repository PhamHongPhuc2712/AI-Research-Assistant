"""
ingestion.py — Document ingestion pipeline.

Loads PDF or text files, splits them into chunks, embeds them using OpenAI
embeddings, and stores the resulting vectors in a ChromaDB collection.

Exposes a single public function: ingest_document(file_path) -> dict.
"""

import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from backend.config import settings

logger = logging.getLogger(__name__)

# Supported file extensions and their corresponding loaders.
_LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def _get_loader(file_path: Path):
    """Return the appropriate LangChain document loader for a given file.

    Args:
        file_path: Path to the file to load.

    Returns:
        An instantiated document loader ready to call .load().

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = file_path.suffix.lower()
    loader_cls = _LOADER_MAP.get(suffix)

    if loader_cls is None:
        supported = ", ".join(_LOADER_MAP.keys())
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported types: {supported}"
        )

    return loader_cls(str(file_path))


def _split_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def _store_in_vectordb(chunks: list, persist_directory: str) -> None:
    """Embed chunks and store them in ChromaDB.

    Args:
        chunks: List of chunked Document objects to embed and store.
        persist_directory: Filesystem path for ChromaDB persistent storage.

    Raises:
        RuntimeError: If embedding or storage fails.
    """
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    )
    db.add_documents(chunks)


def ingest_document(file_path: str) -> dict:
    """Ingest a single document into the vector store.

    Loads the file, splits it into chunks based on configured CHUNK_SIZE and
    CHUNK_OVERLAP, embeds each chunk using OpenAI embeddings, and stores the
    results in ChromaDB.

    Args:
        file_path: Absolute or relative path to a PDF, TXT, or MD file.

    Returns:
        A dict with metadata about the ingestion:
            - "file": the basename of the ingested file.
            - "chunks_stored": the number of chunks written to the vector store.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
        RuntimeError: If embedding or vector store operations fail.
    """
    path = Path(file_path)

    # --- Validate the file exists ---
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    logger.info("Starting ingestion for '%s'", path.name)

    # --- Load ---
    try:
        loader = _get_loader(path)
        documents = loader.load()
    except ValueError:
        raise  # Re-raise unsupported file type as-is.
    except Exception as exc:
        raise RuntimeError(f"Failed to load file '{path.name}': {exc}") from exc

    if not documents:
        logger.warning("No content extracted from '%s'.", path.name)
        return {"file": path.name, "chunks_stored": 0}

    # --- Chunk ---
    try:
        chunks = _split_documents(
            documents,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to split '{path.name}' into chunks: {exc}"
        ) from exc

    logger.info("Split '%s' into %d chunks.", path.name, len(chunks))

    # --- Embed & Store ---
    try:
        _store_in_vectordb(chunks, persist_directory=settings.CHROMA_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to embed/store chunks from '{path.name}': {exc}"
        ) from exc

    logger.info(
        "Successfully stored %d chunks from '%s' into ChromaDB.",
        len(chunks),
        path.name,
    )

    return {"file": path.name, "chunks_stored": len(chunks)}
