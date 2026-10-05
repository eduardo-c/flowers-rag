"""FastAPI: orquesta ingesta, consulta y estado. Única capa que toca Chroma y Google AI."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import HealthStatus
from app.settings import get_settings
from app.store import ping

API_VERSION = "1.0.0"

settings = get_settings()

app = FastAPI(
    title="Sistema RAG — Catálogo de Flores y Listas de Precios",
    version=API_VERSION,
    description="API del sistema RAG: ingesta con embeddings de Google AI y consulta sobre ChromaDB.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthStatus,
    tags=["salud"],
    operation_id="healthCheck",
    summary="Verifica el estado de la API, de ChromaDB y de la clave de Google AI.",
)
def health() -> HealthStatus:
    """Siempre 200: si Chroma no responde o falta la clave, `status: degraded`."""
    current = get_settings()
    chroma_accessible = ping(current)

    chunks = 0
    if chroma_accessible:
        from app.store import count_chunks

        try:
            chunks = count_chunks(current)
        except Exception:
            chroma_accessible = False

    key_configured = current.key_configured
    return HealthStatus(
        status="ok" if chroma_accessible and key_configured else "degraded",
        version=API_VERSION,
        chroma_accessible=chroma_accessible,
        google_ai_key_configured=key_configured,
        collection=current.chroma_collection,
        chunks_in_index=chunks,
        embedding_model=current.embedding_model,
        generation_model=current.generation_model,
    )