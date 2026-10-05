"""Modelos de datos. Reflejan `components.schemas` de `docs/openapi_spec.yaml`
y los tipos internos descritos en `x-modulos`."""

from __future__ import annotations

from typing import Literal

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


class IngestRequest(BaseModel):
    """Esquema `IngestRequest` del spec: ingesta por rutas del servidor (modo JSON).

    `paths` es opcional para que una petición sin entrada llegue al endpoint y
    responda `400 ingest_no_input`, en lugar de un `422` de validación genérico.
    """

    paths: list[str] = Field(
        default_factory=list,
        description="Rutas de archivo o carpeta, relativas a la raíz del proyecto (o absolutas).",
        examples=[["data/"]],
    )
    chunk_size: int | None = Field(
        default=None, ge=100, le=1000, description="Palabras por chunk; si falta, `CHUNK_SIZE`."
    )
    chunk_overlap: int | None = Field(
        default=None, ge=0, description="Palabras de solape; si falta, `CHUNK_OVERLAP`."
    )
    reset: bool = Field(default=False, description="Si `true`, vacía la colección antes de indexar.")


class SkippedFile(BaseModel):
    """Esquema `SkippedFile` del spec: archivo omitido y motivo."""

    source: str
    reason: str


class IngestResponse(BaseModel):
    """Esquema `IngestResponse` del spec."""

    documents_processed: int = Field(ge=0)
    chunks_indexed: int = Field(ge=0)
    chunks_skipped: int = Field(default=0, ge=0)
    collection: str
    embedding_model: str
    sources: list[str]
    skipped: list[SkippedFile] = Field(default_factory=list)
    duration_ms: int = Field(ge=0)


class QueryRequest(BaseModel):
    """Esquema `QueryRequest` del spec.

    `question` no lleva `min_length` para que una pregunta vacía o de solo espacios
    llegue al endpoint y responda `400 empty_question`, y no un `422` genérico.
    """

    question: str = Field(max_length=2000, description="Pregunta en lenguaje natural.")
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="Vecinos a recuperar; si falta, `TOP_K_DEFAULT`."
    )
    source: str | None = Field(
        default=None, description="Filtro opcional por documento (reto opcional)."
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Umbral de similitud coseno; si falta, `MIN_SCORE_DEFAULT`.",
    )


class QueryResponse(BaseModel):
    """Esquema `QueryResponse` del spec: respuesta RAG o abstención, ambas `200`.

    `citations` van numeradas desde 1 y ese mismo número es el `[n]` que aparece en
    `answer` (regla 03).
    """

    answer: str
    abstained: bool
    abstain_reason: Literal["sin_evidencia", "modelo_abstuvo"] | None = None
    citations: list[RetrievedChunk] = Field(default_factory=list)
    question_embedding_model: str
    generation_model: str | None = None
    top_k: int = Field(ge=1)
    retrieval_ms: int = Field(ge=0)
    generation_ms: int = Field(ge=0)