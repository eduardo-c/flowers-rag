"""Pruebas de `POST /ingest` contra el contrato de `docs/openapi_spec.yaml`.

Se indexa un archivo real de `/data`. Los embeddings los produce el código real de
`app/embed.py`, pero con el doble de `google.genai.Client`: sin red y sin cuota.
"""

from __future__ import annotations

import pytest

from tests.fakes import FakeClient

pytestmark = pytest.mark.anyio

DOCUMENTO = "data/01_cuidados_y_temporadas_rosas.md"
DIMENSION = 768


@pytest.fixture
def embeddings_falsos(monkeypatch):
    """Sustituye el cliente de Google AI: captura los lotes que se envían."""
    client = FakeClient(dim=DIMENSION)
    monkeypatch.setattr("app.embed.genai.Client", lambda **_: client)
    return client


# --- Caso real: un documento de /data -------------------------------------------


async def test_ingest_de_un_archivo_de_data(api, embeddings_falsos):
    """El endpoint lee un archivo de `/data`, lo trocea y persiste sus vectores."""
    response = await api.post("/ingest", json={"paths": [DOCUMENTO]})

    assert response.status_code == 200, response.text
    cuerpo = response.json()
    assert cuerpo["documents_processed"] == 1
    assert cuerpo["chunks_indexed"] >= 2
    assert cuerpo["sources"] == [DOCUMENTO]
    assert cuerpo["skipped"] == []
    from app.settings import get_settings

    assert cuerpo["embedding_model"] == get_settings().embedding_model
    assert cuerpo["collection"] == "test_flowers_catalog"
    assert cuerpo["duration_ms"] >= 0


async def test_los_chunks_quedan_en_chroma_con_su_solape(api, embeddings_falsos):
    """Regla 03: los vectores los calculó Google AI y los chunks conservan el solape."""
    from app.settings import get_settings
    from app.store import get_collection

    await api.post("/ingest", json={"paths": [DOCUMENTO], "chunk_size": 120, "chunk_overlap": 40})
    guardados = get_collection(get_settings()).get(include=["metadatas", "documents"])

    assert guardados["metadatas"], "no se guardó ningún chunk"
    assert all(m["source"] == DOCUMENTO for m in guardados["metadatas"])
    assert [m["chunk_index"] for m in guardados["metadatas"]] == sorted(
        m["chunk_index"] for m in guardados["metadatas"]
    )
    assert all(len(d.split()) <= 120 for d in guardados["documents"])

    # El solape declarado se respeta: el comienzo del segundo chunk ya está en el primero.
    assert guardados["documents"][1].split()[0] in guardados["documents"][0].split()


async def test_ingest_reutiliza_el_embedder_de_google_ai(api, embeddings_falsos):
    """Nada de embeddings locales: los vectores salen del cliente de Google AI."""
    from app.settings import get_settings

    await api.post("/ingest", json={"paths": [DOCUMENTO]})

    assert embeddings_falsos.calls, "hubo que llamar a la API de embeddings"
    assert {call["model"] for call in embeddings_falsos.calls} == {get_settings().embedding_model}


async def test_health_refleja_el_indice_tras_ingerir(api, embeddings_falsos):
    await api.post("/ingest", json={"paths": [DOCUMENTO]})

    cuerpo = (await api.get("/health")).json()

    assert cuerpo["chunks_in_index"] >= 2
    assert cuerpo["status"] == "ok"


# --- Errores declarados en el spec ---------------------------------------------


async def test_sin_rutas_ni_archivos_da_400(api, embeddings_falsos):
    response = await api.post("/ingest", json={})

    assert response.status_code == 400
    assert response.json()["code"] == "ingest_no_input"


async def test_sin_clave_de_google_ai_da_503(api, embeddings_falsos, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    response = await api.post("/ingest", json={"paths": [DOCUMENTO]})

    assert response.status_code == 503
    assert response.json()["code"] == "missing_google_api_key"


async def test_un_archivo_ilegible_se_reporta_en_skipped(api, embeddings_falsos):
    """La ingesta no es atómica: lo que falla se reporta, no tumba el resto."""
    response = await api.post("/ingest", json={"paths": ["data/no-existe.md"]})

    assert response.status_code == 200, response.text
    cuerpo = response.json()
    assert cuerpo["documents_processed"] == 0
    assert cuerpo["chunks_indexed"] == 0
    assert cuerpo["skipped"] == [{"source": "data/no-existe.md", "reason": "ilegible"}]


async def test_un_formato_no_soportado_se_reporta_en_skipped(api, embeddings_falsos, tmp_path):
    (tmp_path / "notas.docx").write_bytes(b"PK\x03\x04")

    response = await api.post("/ingest", json={"paths": [str(tmp_path / "notas.docx")]})

    assert response.status_code == 200, response.text
    assert response.json()["skipped"][0]["reason"] == "formato_no_soportado"


async def test_chunk_size_fuera_de_rango_da_422(api, embeddings_falsos):
    """El spec limita `chunk_size` a 100–1000 palabras."""
    response = await api.post(
        "/ingest", json={"paths": [DOCUMENTO], "chunk_size": 10}
    )

    assert response.status_code == 422