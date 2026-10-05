"""Configuración por entorno. Contrato: `x-configuracion-entorno` del spec.

Se lee en cada llamada a `get_settings()` (y no al importar) para que las pruebas
puedan cambiar el entorno con `monkeypatch` sin recargar módulos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in _str(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    embedding_model: str
    generation_model: str
    chroma_path: str
    chroma_collection: str
    chunk_size: int
    chunk_overlap: int
    top_k_default: int
    min_score_default: float
    max_file_bytes: int
    embed_batch_size: int
    google_timeout_seconds: int
    allowed_origins: list[str]

    @property
    def key_configured(self) -> bool:
        return bool(self.google_api_key)


def get_settings() -> Settings:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    return Settings(
        google_api_key=key or None,
        embedding_model=_str("EMBEDDING_MODEL", "text-embedding-004"),
        generation_model=_str("GENERATION_MODEL", "gemini-2.0-flash"),
        chroma_path=_str("CHROMA_PATH", "./chroma"),
        chroma_collection=_str("CHROMA_COLLECTION", "flowers_catalog"),
        chunk_size=_int("CHUNK_SIZE", 320),
        chunk_overlap=_int("CHUNK_OVERLAP", 64),
        top_k_default=_int("TOP_K_DEFAULT", 3),
        min_score_default=_float("MIN_SCORE_DEFAULT", 0.35),
        max_file_bytes=_int("MAX_FILE_BYTES", 25 * 1024 * 1024),
        embed_batch_size=_int("EMBED_BATCH_SIZE", 16),
        google_timeout_seconds=_int("GOOGLE_TIMEOUT_SECONDS", 30),
        allowed_origins=_list(
            "ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
        ),
    )