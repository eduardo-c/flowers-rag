"""Pruebas unitarias de `app/embed.py`.

Contrato: sección `x-modulos -> app/embed.py` de `docs/openapi_spec.yaml`.
Criterio de la regla 03: los vectores salen **solo** de Google AI.
Sin red: se inyecta un doble de `google.genai.Client`.
"""

from __future__ import annotations

import pytest
from google.genai import errors as genai_errors

from tests.fakes import FakeClient, fake_client, fake_client_factory, imports_de, vector

pytestmark = pytest.mark.anyio


@pytest.fixture
def embedder(fake_client, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "clave-de-prueba")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-004")
    from app.embed import GoogleAIEmbedder

    return GoogleAIEmbedder(client=fake_client)


# --- Rúbrica 9-10: mismo modelo para documentos y preguntas --------------------


async def test_usa_el_modelo_configurado(embedder, fake_client):
    await embedder.embed_documents(["rosas"])

    assert [call["model"] for call in fake_client.calls] == ["text-embedding-004"]


async def test_documentos_y_preguntas_usan_el_mismo_modelo(embedder, fake_client):
    await embedder.embed_documents(["rosas", "tulipanes"])
    await embedder.embed_query("¿cuánto cuestan las rosas?")

    assert {call["model"] for call in fake_client.calls} == {"text-embedding-004"}


# --- Contrato de salida --------------------------------------------------------


async def test_devuelve_un_vector_por_texto(embedder):
    vectors = await embedder.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(isinstance(v, list) and len(v) == 8 for v in vectors)
    assert vectors[0] == vector(0)
    assert vectors[2] == vector(2)


async def test_todos_los_valores_son_float(embedder):
    (first,) = await embedder.embed_documents(["solo una"])

    assert all(type(value) is float for value in first)
    assert embedder.dimension == 8


async def test_embed_query_devuelve_un_solo_vector(embedder, fake_client):
    result = await embedder.embed_query("¿precio de las rosas?")

    assert isinstance(result, list)
    assert len(result) == 8
    assert len(fake_client.calls) == 1


async def test_lista_vacia_no_llama_a_la_api(embedder, fake_client):
    assert await embedder.embed_documents([]) == []
    assert fake_client.calls == []


# --- Cuota: lotes pequeños -----------------------------------------------------


async def test_respeta_embed_batch_size(fake_client, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "clave-de-prueba")
    monkeypatch.setenv("EMBED_BATCH_SIZE", "2")
    from app.embed import GoogleAIEmbedder

    embedder = GoogleAIEmbedder(client=fake_client)

    vectors = await embedder.embed_documents(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert [len(call["contents"]) for call in fake_client.calls] == [2, 2, 1]


# --- Errores: nunca hay fallback silencioso ------------------------------------


async def test_sin_clave_lanza_embedding_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from app.embed import EmbeddingError, GoogleAIEmbedder

    embedder = GoogleAIEmbedder(client=None)

    with pytest.raises(EmbeddingError) as excinfo:
        await embedder.embed_documents(["texto"])
    assert excinfo.value.code == "missing_google_api_key"


async def test_error_de_la_api_se_traduce_a_embedding_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "clave-de-prueba")
    from app.embed import EmbeddingError, GoogleAIEmbedder

    cliente = FakeClient(
        error=genai_errors.APIError(code=429, response_json={"error": {"message": "resource_exhausted"}})
    )

    with pytest.raises(EmbeddingError) as excinfo:
        await GoogleAIEmbedder(client=cliente).embed_query("precio")
    assert excinfo.value.code == "google_ai_error"


async def test_respuesta_inconsistente_lanza_embedding_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "clave-de-prueba")
    from app.embed import EmbeddingError, GoogleAIEmbedder

    with pytest.raises(EmbeddingError) as excinfo:
        await GoogleAIEmbedder(client=FakeClient(mismatch=True)).embed_documents(["a", "b"])
    assert excinfo.value.code == "invalid_response"


async def test_cambio_de_dimension_lanza_embedding_error(fake_client, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "clave-de-prueba")
    from app.embed import EmbeddingError, GoogleAIEmbedder

    embedder = GoogleAIEmbedder(client=fake_client)
    await embedder.embed_documents(["a"])
    fake_client.models.dim = 16

    with pytest.raises(EmbeddingError) as excinfo:
        await embedder.embed_documents(["b"])
    assert excinfo.value.code == "invalid_response"


async def test_texto_vacio_no_se_vectoriza(embedder, fake_client):
    with pytest.raises(ValueError):
        await embedder.embed_query("   ")
    assert fake_client.calls == []


# --- Cableado real con el SDK (regla 03: google-genai) ------------------------


async def test_construye_el_cliente_real_con_la_clave_y_el_timeout(fake_client_factory, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "clave-de-prueba")
    monkeypatch.setenv("GOOGLE_TIMEOUT_SECONDS", "12")
    from app.embed import GoogleAIEmbedder

    await GoogleAIEmbedder().embed_query("hola")

    (kwargs,) = fake_client_factory
    assert kwargs["api_key"] == "clave-de-prueba"
    assert kwargs["http_options"].timeout == 12000  # milisegundos


def test_los_embeddings_salen_de_google_genai():
    """Regla 03: `google-genai` es el único motor de embeddings."""
    import app.embed as embed_module

    modulos = imports_de(embed_module.__file__)

    assert "google.genai" in modulos
    assert not [m for m in modulos if m.split(".")[0] == "chromadb"], "el índice no se toca desde embed.py"


def test_no_hay_backend_de_embeddings_local():
    """Regla 03: ni FastText, ni BERT local, ni bolsa de palabras."""
    import app.embed as embed_module

    modulos = imports_de(embed_module.__file__)
    prohibidos = {"sentence_transformers", "fastembed", "sklearn", "gensim", "fasttext", "torch", "transformers"}

    assert not prohibidos & {m.split(".")[0] for m in modulos}, modulos