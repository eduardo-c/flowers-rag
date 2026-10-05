"""Pruebas unitarias de `app/store.py`.

Contrato: sección `x-modulos -> app/store.py` de `docs/openapi_spec.yaml`.
Criterios de la regla 03 que se comprueban aquí:
  1. Los embeddings los da Google AI: se pasa SIEMPRE `embedding_function=None`
     y `add_chunks` exige los vectores ya calculados.
  4. Persistencia: `PersistentClient` sobre `./chroma`; el índice sobrevive.
Sin red y sin Google AI: ChromaDB real en un directorio temporal.
"""

from __future__ import annotations

import dataclasses

import chromadb
import pytest

from app.settings import get_settings
from tests.fakes import imports_de, vector

pytestmark = pytest.mark.anyio


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "test_flowers_catalog")
    return get_settings()


@pytest.fixture
def chunks():
    from app.models import Chunk

    return [
        Chunk(source="data/rosas.md", text="Ramo de 20 rosas: 89,90 €", chunk_index=0, title="Rosas"),
        Chunk(source="data/rosas.md", text="Las rosas duran 5 días en agua", chunk_index=1, title="Rosas"),
        Chunk(source="data/iris.md", text="El iris se riega cada 3 días", chunk_index=0, title="Iris", page=2),
    ]


class ClienteEspia:
    """Doble de `PersistentClient`: registra cómo se construye la colección."""

    def __init__(self) -> None:
        self.llamadas: list[dict] = []

    def get_or_create_collection(self, name, **kwargs):
        self.llamadas.append({"name": name, **kwargs})
        return object()


@pytest.fixture
def cliente_espia(monkeypatch):
    espia = ClienteEspia()
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **kwargs: espia)
    return espia


# --- Rúbrica 6: PersistentClient sobre ./chroma --------------------------------


def test_get_client_devuelve_un_cliente_persistente(settings, monkeypatch):
    import app.store as store

    rutas: list[str] = []
    original = chromadb.PersistentClient

    def spy(path, **kwargs):
        rutas.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(chromadb, "PersistentClient", spy)
    store.reset_clients()

    cliente = store.get_client(settings)

    assert isinstance(cliente, chromadb.api.ClientAPI)
    assert rutas == [settings.chroma_path], "la ruta debe ser CHROMA_PATH (./chroma por defecto)"


def test_la_ruta_por_defecto_es_chroma(monkeypatch):
    monkeypatch.delenv("CHROMA_PATH", raising=False)
    monkeypatch.delenv("CHROMA_COLLECTION", raising=False)

    ajustes = get_settings()

    assert ajustes.chroma_path == "./chroma"
    assert ajustes.chroma_collection == "flowers_catalog"


# --- Regla 03 / 1: nunca el embedder por defecto de Chroma ---------------------


def test_la_coleccion_se_crea_sin_funcion_de_embeddings(settings, cliente_espia):
    import app.store as store

    store.reset_clients()
    store.get_collection(settings)

    (llamada,) = cliente_espia.llamadas
    assert llamada["name"] == "test_flowers_catalog"
    assert llamada["embedding_function"] is None
    assert llamada["metadata"]["hnsw:space"] == "cosine"


# --- add_chunks: los embeddings entran desde fuera -----------------------------


async def test_add_chunks_exige_el_parametro_embeddings(settings, chunks):
    import app.store as store

    with pytest.raises(TypeError):
        await store.add_chunks(settings, chunks)  # type: ignore[call-arg]


async def test_add_chunks_rechaza_embeddings_vacios(settings, chunks):
    import app.store as store
    from app.store import StoreError

    with pytest.raises(StoreError) as excinfo:
        await store.add_chunks(settings, chunks, [])
    assert excinfo.value.code == "embeddings_missing"


async def test_add_chunks_exige_un_vector_por_chunk(settings, chunks):
    import app.store as store
    from app.store import StoreError

    with pytest.raises(StoreError) as excinfo:
        await store.add_chunks(settings, chunks, [vector(0)])
    assert excinfo.value.code == "embeddings_count_mismatch"


async def test_add_chunks_rechaza_dimensiones_distintas(settings, chunks):
    import app.store as store
    from app.store import StoreError

    with pytest.raises(StoreError) as excinfo:
        await store.add_chunks(settings, chunks, [vector(0), vector(1, 4), vector(2)])
    assert excinfo.value.code == "embeddings_ragged"


async def test_add_chunks_guarda_exactamente_los_embeddings_de_google_ai(settings, chunks):
    """Si el vector guardado es el que pasamos, nadie lo ha recalculado por su cuenta."""
    import app.store as store

    embeddings = [vector(0), vector(1), vector(2)]

    ids = await store.add_chunks(settings, chunks, embeddings)

    assert ids == ["data/rosas.md#chunk-000", "data/rosas.md#chunk-001", "data/iris.md#chunk-000"]
    guardados = store.get_collection(settings).get(ids=ids, include=["embeddings"])
    for guardada, esperada in zip(guardados["embeddings"], embeddings):
        assert list(map(float, guardada)) == pytest.approx(esperada, abs=1e-6)


async def test_add_chunks_omite_metadatos_nulos(settings, chunks):
    import app.store as store

    ids = await store.add_chunks(settings, chunks, [vector(0), vector(1), vector(2)])

    metadatos = store.get_collection(settings).get(ids=ids, include=["metadatas"])["metadatas"]
    assert metadatos[0] == {"source": "data/rosas.md", "title": "Rosas", "chunk_index": 0}
    assert metadatos[2] == {"source": "data/iris.md", "title": "Iris", "chunk_index": 0, "page": 2}


async def test_add_chunks_devuelve_cero_sin_chunks(settings):
    import app.store as store

    assert await store.add_chunks(settings, [], []) == []


# --- query_top_k: k-NN por coseno con el vector de la pregunta -----------------


@pytest.fixture
async def indexado(settings, chunks):
    import app.store as store

    await store.add_chunks(settings, chunks, [vector(0), vector(1), vector(2)])
    return settings


async def test_query_top_k_devuelve_score_uno_menos_distancia(indexado):
    import app.store as store

    resultados = await store.query_top_k(indexado, vector(0), top_k=2)

    assert [r.index for r in resultados] == [1, 2]
    assert resultados[0].id == "data/rosas.md#chunk-000"
    assert resultados[0].source == "data/rosas.md"
    assert resultados[0].text.startswith("Ramo de 20 rosas")
    assert resultados[0].title == "Rosas"
    assert resultados[0].page is None
    assert resultados[0].score == pytest.approx(1.0, abs=1e-4)
    assert resultados[0].score >= resultados[1].score


async def test_query_top_k_pide_el_embedding_por_parametro(indexado):
    import app.store as store

    with pytest.raises(TypeError):
        await store.query_top_k(indexado, top_k=2)  # type: ignore[call-arg]


async def test_query_top_k_filtra_por_source(indexado):
    import app.store as store

    resultados = await store.query_top_k(indexado, vector(0), top_k=3, source="data/iris.md")

    assert [r.source for r in resultados] == ["data/iris.md"]
    assert resultados[0].page == 2


async def test_query_top_k_rechaza_dimension_distinta(indexado):
    import app.store as store
    import app.store as store
    from app.store import StoreError

    with pytest.raises(StoreError) as excinfo:
        await store.query_top_k(indexado, vector(0, 4), top_k=2)
    assert excinfo.value.code == "dimension_mismatch"


async def test_query_top_k_valida_top_k(indexado):
    import app.store as store

    with pytest.raises(ValueError):
        await store.query_top_k(indexado, vector(0), top_k=0)


async def test_query_top_k_sin_coincidencias_devuelve_lista_vacia(indexado):
    import app.store as store

    assert await store.query_top_k(indexado, vector(0), top_k=3, source="data/no-existe.md") == []


# --- Rúbrica 6: persistencia real en disco -------------------------------------


async def test_el_indice_sobrevive_a_un_reinicio(settings, chunks, tmp_path):
    """Limpiar la caché de clientes simula reiniciar FastAPI: los datos siguen ahí."""
    import app.store as store

    await store.add_chunks(settings, chunks, [vector(0), vector(1), vector(2)])
    cliente_antes = store.get_client(settings)

    store.reset_clients()
    cliente_despues = store.get_client(settings)

    assert cliente_despues is not cliente_antes
    assert store.count_chunks(settings) == 3
    assert (await store.query_top_k(settings, vector(0), top_k=1))[0].source == "data/rosas.md"
    assert (tmp_path / "chroma").exists()


async def test_count_chunks_cero_si_no_existe_coleccion(settings):
    import app.store as store

    assert store.count_chunks(settings) == 0


async def test_delete_source_solo_borra_ese_documento(settings, chunks):
    import app.store as store

    await store.add_chunks(settings, chunks, [vector(0), vector(1), vector(2)])

    borrados = await store.delete_source(settings, "data/rosas.md")

    assert borrados == 2
    assert store.count_chunks(settings) == 1
    assert await store.delete_source(settings, "data/rosas.md") == 0


def test_ping_false_si_chroma_no_es_accesible(settings, tmp_path):
    import app.store as store

    bloqueado = tmp_path / "bloqueado"
    bloqueado.write_text("archivo, no directorio")
    ajustes = dataclasses.replace(settings, chroma_path=str(bloqueado / "chroma"))
    store.reset_clients()

    assert store.ping(ajustes) is False


def test_ping_true_con_chroma_accesible(settings):
    import app.store as store

    assert store.ping(settings) is True


# --- Regla 03: store.py no habla con Google AI ni genera texto ----------------


def test_store_no_depende_de_google_ai():
    import app.store as store

    modulos = imports_de(store.__file__)

    assert not [m for m in modulos if m.split(".")[0] == "google"], modulos