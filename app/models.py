"""Modelos de datos. Reflejan `components.schemas` de `docs/openapi_spec.yaml`
y los tipos internos descritos en `x-modulos`."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    """Esquema `HealthStatus` del spec."""

    status: str = Field(description="`ok` = Chroma responde y hay clave.")
    version: str
    chroma_accessible: bool
    google_ai_key_configured: bool
    collection: str
    chunks_in_index: int = Field(ge=0)
    embedding_model: str
    generation_model: str


class Chunk(BaseModel):
    """Trozo indexable. Sus metadatos van a Chroma; los `None` se omiten."""

    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    title: str | None = None
    page: int | None = Field(default=None, ge=1)


class RetrievedChunk(BaseModel):
    """Vecino del k-NN. Se serializa tal cual en `citations[]` de `POST /query`.

    `score` es la similitud coseno (`1 - distancia`) y `index` es 1-based: coincide
    con el `[n]` que Gemini escribe en la respuesta.
    """

    index: int = Field(ge=1)
    id: str
    source: str
    text: str
    score: float
    title: str | None = None
    chunk_index: int | None = None
    page: int | None = None