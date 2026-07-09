from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    from langchain_community.vectorstores import Chroma

try:
    from langchain_ollama import OllamaEmbeddings
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    from langchain_community.embeddings import OllamaEmbeddings


def build_ollama_embeddings(model_name: str = "mxbai-embed-large:335m", host: str | None = None) -> OllamaEmbeddings:
    kwargs: dict[str, Any] = {"model": model_name}
    if host:
        kwargs["base_url"] = host
    return OllamaEmbeddings(**kwargs)


def build_chroma_from_documents(
    documents: Sequence[Document],
    *,
    collection_name: str,
    persist_directory: str | Path,
    embedding_model: str = "mxbai-embed-large:335m",
    ollama_host: str | None = None,
    ids: Sequence[str] | None = None,
) -> Chroma:
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = build_ollama_embeddings(embedding_model, ollama_host)
    return Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_path),
        ids=list(ids) if ids is not None else None,
    )


def load_chroma_documents(
    *, persist_directory: str | Path, collection_name: str
) -> list[tuple[str, str, dict[str, Any]]]:
    import chromadb

    persist_path = Path(persist_directory)
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_collection(name=collection_name)
    payload = collection.get(include=["documents", "metadatas"])

    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []

    rows: list[tuple[str, str, dict[str, Any]]] = []
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        if not isinstance(document, str):
            continue
        rows.append((str(chunk_id), document, dict(metadata or {})))
    return rows
