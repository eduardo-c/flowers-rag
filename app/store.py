"""ChromaDB: único acceso al índice vectorial (`./chroma`).

Los embeddings los calcula Google AI en `app/embed.py`; aquí nunca se usa el
embedder por defecto de ChromaDB, siempre se pasan `embeddings=` explícitos.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.settings import Settings

_clients: dict[tuple[str, str], chromadb.ClientAPI] = {}


def get_client(settings: Settings) -> chromadb.ClientAPI:
    """`PersistentClient` cacheado por (ruta, nombre) para no reabrir en cada request."""
    key = (settings.chroma_path, settings.chroma_collection)
    if key not in _clients:
        _clients[key] = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
    return _clients[key]


def count_chunks(settings: Settings) -> int:
    """Chunks persistidos. `0` si la colección todavía no existe."""
    client = get_client(settings)
    collections = client.list_collections()
    names = {getattr(item, "name", item) for item in collections}
    if settings.chroma_collection not in names:
        return 0
    return client.get_collection(settings.chroma_collection).count()


def ping(settings: Settings) -> bool:
    """True si ChromaDB responde; False si la ruta no es utilizable."""
    try:
        count_chunks(settings)
    except Exception:
        return False
    return True