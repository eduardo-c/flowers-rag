"""ChromaDB: único acceso al índice vectorial.

Regla de oro del proyecto: **los embeddings los calcula Google AI en
`app/embed.py` y entran aquí como parámetro obligatorio**. La colección se crea
siempre con `embedding_function=None` para no activar el embedder por defecto de
ChromaDB (all-MiniLM), que está prohibido en este proyecto.

Este módulo no llama a Google AI y no genera texto.

El cliente de Chroma es bloqueante: las operaciones de datos son `async` y
delegan el trabajo real a un hilo (`anyio.to_thread`) para no bloquear el event
loop. `ping` y `count_chunks` son síncronas porque las usa el endpoint síncrono
`GET /health`.
"""

from __future__ import annotations

from typing import Sequence

import anyio.to_thread
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models import Chunk, RetrievedChunk
from app.settings import Settings

_clients: dict[tuple[str, str], chromadb.ClientAPI] = {}


class StoreError(RuntimeError):
    """Falla al hablar con el índice. `code` es estable (ver `x-modulos` del spec)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def reset_clients() -> None:
    """Olvida los clientes cacheados. Lo usa el reinicio de FastAPI y las pruebas."""
    _clients.clear()


def get_client(settings: Settings) -> chromadb.ClientAPI:
    """`PersistentClient` sobre `CHROMA_PATH` (`./chroma` por defecto).

    Cacheado por (ruta, colección): el índice vive en disco, así que reiniciar el
    proceso no lo borra.
    """
    key = (settings.chroma_path, settings.chroma_collection)
    if key not in _clients:
        _clients[key] = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
    return _clients[key]


def get_collection(settings: Settings, create: bool = True):
    """Colección en coseno y **sin** función de embeddings de Chroma."""
    client = get_client(settings)
    if create:
        return client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
    return client.get_collection(
        name=settings.chroma_collection,
        embedding_function=None,
    )


def chunk_id(chunk: Chunk) -> str:
    """Id estable: `data/precios.csv#chunk-004` (ver `x-modulos` del spec)."""
    return f"{chunk.source}#chunk-{chunk.chunk_index:03d}"


async def add_chunks(
    settings: Settings,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
) -> list[str]:
    """Guarda chunks + vectores ya calculados por Google AI. Devuelve los ids.

    `embeddings` es obligatorio: sin vectores de Google AI no se indexa nada.
    """
    vectors = _validated_embeddings(chunks, embeddings)
    if not chunks:
        return []
    return await anyio.to_thread.run_sync(_add, settings, list(chunks), vectors)


async def query_top_k(
    settings: Settings,
    embedding: Sequence[float],
    top_k: int,
    *,
    source: str | None = None,
) -> list[RetrievedChunk]:
    """k-NN por coseno con el embedding de la pregunta (Google AI).

    `score` es la similitud (`1 - distancia`) y `index` es 1-based: el mismo `[n]`
    que escribe Gemini en la respuesta.
    """
    if top_k < 1:
        raise ValueError("top_k debe ser >= 1.")
    vector = list(map(float, embedding))
    if not vector:
        raise StoreError("embeddings_missing", "El vector de la pregunta está vacío.")
    return await anyio.to_thread.run_sync(_query, settings, vector, top_k, source)


async def delete_source(settings: Settings, source: str) -> int:
    """Borra los chunks de un documento por metadato. Idempotente."""
    return await anyio.to_thread.run_sync(_delete, settings, source)


def count_chunks(settings: Settings) -> int:
    """Chunks persistidos; `0` si la colección todavía no existe. Síncrono (`/health`)."""
    client = get_client(settings)
    nombres = {getattr(item, "name", item) for item in client.list_collections()}
    if settings.chroma_collection not in nombres:
        return 0
    return client.get_collection(settings.chroma_collection).count()


def ping(settings: Settings) -> bool:
    """True si ChromaDB responde; False si la ruta no es utilizable. No propaga."""
    try:
        count_chunks(settings)
    except Exception:
        return False
    return True


# --- validación (sin E/S: falla rápido y con mensaje claro) ----------------------


def _validated_embeddings(
    chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
) -> list[list[float]]:
    if not chunks:
        return []
    if not embeddings:
        raise StoreError(
            "embeddings_missing",
            "Los embeddings de Google AI son obligatorios: no se puede indexar sin vectores.",
        )
    if len(chunks) != len(embeddings):
        raise StoreError(
            "embeddings_count_mismatch",
            f"{len(chunks)} chunks pero {len(embeddings)} embeddings: debe haber un vector por chunk.",
        )

    dimension = len(embeddings[0])
    if dimension == 0:
        raise StoreError("embeddings_missing", "El vector de Google AI está vacío.")
    if any(len(vector) != dimension for vector in embeddings):
        raise StoreError(
            "embeddings_ragged",
            f"Los embeddings de Google AI no tienen la misma dimensión: se esperaba {dimension}.",
        )
    return [list(map(float, vector)) for vector in embeddings]


# --- operaciones bloqueantes (se ejecutan en un hilo) ---------------------------


def _add(settings: Settings, chunks: list[Chunk], vectors: list[list[float]]) -> list[str]:
    ids = [chunk_id(chunk) for chunk in chunks]
    get_collection(settings).add(
        ids=ids,
        documents=[chunk.text for chunk in chunks],
        metadatas=[_metadatos(chunk) for chunk in chunks],
        embeddings=vectors,
    )
    return ids


def _query(
    settings: Settings, embedding: list[float], top_k: int, source: str | None
) -> list[RetrievedChunk]:
    collection = get_collection(settings)
    _check_dimension(collection, embedding)

    resultado = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where={"source": source} if source else None,
        include=["documents", "metadatas", "distances"],
    )

    ids = (resultado.get("ids") or [[]])[0]
    documentos = (resultado.get("documents") or [[]])[0]
    metadatas = (resultado.get("metadatas") or [[]])[0]
    distancias = (resultado.get("distances") or [[]])[0]

    vecinos: list[RetrievedChunk] = []
    for indice, (id_chunk, texto, metadatos, distancia) in enumerate(
        zip(ids, documentos, metadatas, distancias), start=1
    ):
        vecinos.append(
            RetrievedChunk(
                index=indice,
                id=id_chunk,
                source=(metadatos or {}).get("source", ""),
                text=texto or "",
                score=1.0 - float(distancia),
                title=(metadatos or {}).get("title"),
                chunk_index=(metadatos or {}).get("chunk_index"),
                page=(metadatos or {}).get("page"),
            )
        )
    return vecinos


def _delete(settings: Settings, source: str) -> int:
    collection = get_collection(settings)
    ids = collection.get(where={"source": source}, include=[])["ids"] or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def _metadatos(chunk: Chunk) -> dict[str, str | int]:
    """Metadatos para Chroma: sin `None`, que Chroma no acepta."""
    datos: dict[str, str | int] = {"source": chunk.source, "chunk_index": chunk.chunk_index}
    if chunk.title:
        datos["title"] = chunk.title
    if chunk.page is not None:
        datos["page"] = chunk.page
    return datos


def _check_dimension(collection, embedding: list[float]) -> None:
    if collection.count() == 0:
        return

    guardados = collection.get(limit=1, include=["embeddings"]).get("embeddings")
    if guardados is None or len(guardados) == 0:
        return

    esperada = len(guardados[0])
    if len(embedding) != esperada:
        raise StoreError(
            "dimension_mismatch",
            f"El índice tiene vectores de {esperada} dimensiones y la pregunta trae "
            f"{len(embedding)}: hay que reindexar con el mismo modelo de embeddings.",
        )