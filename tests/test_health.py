"""Pruebas de `GET /health` contra el contrato de `docs/openapi_spec.yaml`.

Cada test cita el requisito de la rubrica que cubre. Sin red y sin Google AI:
solo la app ASGI y un ChromaDB temporal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import UI_ORIGIN

pytestmark = pytest.mark.anyio

SPEC_PATH = Path(__file__).resolve().parents[1] / "docs" / "openapi_spec.yaml"

# Claves exigidas por el esquema HealthStatus del spec.
REQUIRED_FIELDS = {
    "status",
    "version",
    "chroma_accessible",
    "google_ai_key_configured",
    "collection",
    "chunks_in_index",
    "embedding_model",
    "generation_model",
}


async def test_health_responde_200_con_todo_el_contrato(api):
    """Rúbrica 1: /health confirma que la API vive."""
    response = await api.get("/health")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert REQUIRED_FIELDS <= set(response.json())


async def test_health_ok_con_chroma_y_clave(api, health):
    """Rúbrica 1-2: estado ok cuando Chroma responde y hay GOOGLE_API_KEY."""
    assert health["status"] == "ok"
    assert health["chroma_accessible"] is True
    assert health["google_ai_key_configured"] is True
    assert health["collection"] == "test_flowers_catalog"
    assert health["embedding_model"] == "gemini-embedding-001"
    assert health["generation_model"].startswith("gemini")
    assert health["version"] == "1.0.0"
    assert health["chunks_in_index"] == 0


async def test_health_no_devuelve_la_clave(health):
    """La clave vive en el entorno del servidor; nunca sale por HTTP."""
    assert "GOOGLE_API_KEY" not in health
    assert "test-key-no-se-usa" not in str(health)


async def test_health_cuenta_los_chunks_persistidos(api, health):
    """Rúbrica 6: el índice persiste; /health refleja los chunks guardados."""
    from app.settings import get_settings
    from app.store import get_client

    settings = get_settings()
    collection = get_client(settings).get_or_create_collection(
        settings.chroma_collection, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=["c0", "c1", "c2"],
        documents=["rosas", "tulipanes", "orquideas"],
        embeddings=[[0.1, 0.2, 0.3], [0.9, 0.1, 0.0], [0.0, 0.5, 0.5]],
        metadatas=[{"source": "data/a.md"}] * 3,
    )

    response = await api.get("/health")

    assert response.status_code == 200
    assert response.json()["chunks_in_index"] == 3
    assert response.json()["status"] == "ok"


async def test_health_degraded_si_falta_la_clave(api, health, monkeypatch):
    """Rúbrica 17 (clave ausente visible en la UI) sin devolver 500."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    body = (await api.get("/health")).json()

    assert body["status"] == "degraded"
    assert body["google_ai_key_configured"] is False
    assert body["chroma_accessible"] is True


async def test_health_degraded_si_chroma_no_es_accesible(api, monkeypatch, tmp_path):
    """Rúbrica 1-2 y 17: Chroma caído se distingue de API caída (200, no 500)."""
    bloqueado = tmp_path / "bloqueado"
    bloqueado.write_text("esto es un archivo, no un directorio")
    monkeypatch.setenv("CHROMA_PATH", str(bloqueado / "chroma"))

    response = await api.get("/health")

    assert response.status_code == 200, "una caída de Chroma no puede ser un 500"
    body = response.json()
    assert body["status"] == "degraded"
    assert body["chroma_accessible"] is False
    assert body["chunks_in_index"] == 0


async def test_health_permite_el_origen_de_streamlit(api, health):
    """Rúbrica 5: la UI en localhost:8501 puede llamar a la API."""
    response = await api.get("/health", headers={"Origin": UI_ORIGIN})

    assert response.headers.get("access-control-allow-origin") == UI_ORIGIN


async def test_health_cumple_el_esquema_del_spec(health):
    """Regla 02: el cuerpo se valida contra el esquema HealthStatus del YAML."""
    yaml = pytest.importorskip("yaml")
    jsonschema = pytest.importorskip("jsonschema")

    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    schema = spec["components"]["schemas"]["HealthStatus"]

    jsonschema.validate(instance=health, schema=schema)
    assert set(schema["required"]) == REQUIRED_FIELDS
    assert health["status"] in schema["properties"]["status"]["enum"]