"""Modelos de respuesta. Reflejan `components.schemas` de `docs/openapi_spec.yaml`."""

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