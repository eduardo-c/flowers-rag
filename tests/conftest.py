"""Fixtures compartidos.

Regla 02: las pruebas se escriben contra el contrato de `docs/openapi_spec.yaml`.
Nada de red y nada de Google AI: ChromaDB real en un directorio temporal y un
cliente `httpx` sobre la app ASGI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_ORIGIN = "http://localhost:8000"
UI_ORIGIN = "http://localhost:8501"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def api(tmp_path, monkeypatch):
    """App FastAPI con entorno controlado: Chroma temporal, clave y CORS de la UI.

    El entorno se fija *antes* de importar `app.main` para que el middleware CORS
    se construya con los valores de test. `httpx.ASGITransport` es asíncrono, de ahí
    el `AsyncClient` (no se usa el `TestClient` de Starlette, que está deprecado
    frente a `httpx2`).
    """
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "test_flowers_catalog")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-no-se-usa")
    monkeypatch.setenv("ALLOWED_ORIGINS", UI_ORIGIN)

    from app.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url=API_ORIGIN, timeout=30.0
    ) as client:
        yield client


@pytest.fixture
async def health(api) -> dict:
    """`GET /health` y su cuerpo JSON, para aserciones directas en cada test."""
    response = await api.get("/health")
    assert response.status_code == 200, response.text
    return response.json()