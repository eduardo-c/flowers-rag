"""Pruebas de `POST /query` contra el contrato de `docs/openapi_spec.yaml`.

Regla 03: lo que se vigila aquí es que **cada `[n]` de la respuesta mapea a
`citations[index - 1]`** y que una pregunta imposible produce una abstención
visible, nunca una respuesta inventada.

Gemini está doblado: los vectores los decide la prueba y el texto generado es una
canned. Ni red ni cuota.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

pytestmark = pytest.mark.anyio

DOCUMENTO = "data/01_cuidados_y_temporadas_rosas.md"
PREGUNTA = "¿Cómo hay que cortar el tallo de una rosa?"
PREGUNTA_IMPOSIBLE = "¿Quién ganó el Mundial de fútbol de 1998 y con qué marcador?"

# Vectores ortogonales para separar "evidencia" de "nada que ver".
VEC_CHUNK = [1.0, 0.0, 0.0, 0.0]
VEC_PREGUNTA = [0.95, 0.05, 0.0, 0.0]
VEC_IMPOSIBLE = [0.0, 0.0, 0.0, 1.0]


class GeminiFalso:
    """Doble de `google.genai.Client`: `embed_content` y `generate_content`."""

    def __init__(self, *, vector, respuesta: str = "Respuesta con evidencia [1].") -> None:
        self._vector = vector
        self._respuesta = respuesta
        # Vector por texto: así una pregunta puede "no parecerse" a los chunks.
        self.por_texto: dict[str, list[float]] = {}
        self.prompt: str | None = None
        self.models = self

    # --- embeddings ---
    def embed_content(self, *, model, contents, config=None):
        textos = contents if isinstance(contents, list) else [contents]
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=list(self.por_texto.get(texto, self._vector)))
                for texto in textos
            ]
        )

    # --- generación ---
    def generate_content(self, *, model, contents, config=None):
        self.prompt = contents
        self.model = model
        return SimpleNamespace(text=self._respuesta)


class GeminiQueFalla(GeminiFalso):
    """Vectores correctos y generación rota: aísla el fallo de Gemini."""

    def __init__(self, *, vector, excepcion: Exception | None = None, texto: str = "") -> None:
        super().__init__(vector=vector, respuesta=texto)
        self._excepcion = excepcion

    def generate_content(self, *, model, contents, config=None):
        self.prompt = contents
        if self._excepcion is not None:
            raise self._excepcion
        return SimpleNamespace(text=self._respuesta)


def _instalar_gemini(monkeypatch, client) -> GeminiFalso:
    """`app.embed.genai` y `app.generate.genai` son el mismo módulo: se parchea una vez."""
    monkeypatch.setattr("app.embed.genai.Client", lambda **_: client)
    return client


@pytest.fixture
def gemini(monkeypatch):
    """Doble de Gemini instalado, con vectores válidos y respuesta canned."""
    return _instalar_gemini(monkeypatch, GeminiFalso(vector=VEC_CHUNK))


async def _indexar_con_vector(api, monkeypatch, vector) -> GeminiFalso:
    """Ingesta un documento real de `/data` y deja el doble listo para la pregunta."""
    client = _instalar_gemini(monkeypatch, GeminiFalso(vector=vector))
    respuesta = await api.post("/ingest", json={"paths": [DOCUMENTO], "chunk_size": 200})
    assert respuesta.status_code == 200, respuesta.text
    return client


# --- Abstención: la pregunta imposible (regla 03) -------------------------------


async def test_pregunta_imposible_se_abstiene_por_score(api, monkeypatch):
    """Ningún chunk se parece a la pregunta -> abstención SIN llamar a Gemini."""
    client = await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)
    client.por_texto[PREGUNTA_IMPOSIBLE] = VEC_IMPOSIBLE  # ortogonal a los chunks

    respuesta = await api.post("/query", json={"question": PREGUNTA_IMPOSIBLE, "top_k": 3})

    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["abstained"] is True
    assert cuerpo["abstain_reason"] == "sin_evidencia"
    assert cuerpo["citations"] == []
    assert cuerpo["generation_model"] is None
    assert cuerpo["generation_ms"] == 0
    assert "evidencia" in cuerpo["answer"].lower()
    assert client.prompt is None, "no se debe generar texto si no hay evidencia"


async def test_pregunta_imposible_se_abstiene_por_modelo(api, monkeypatch):
    """Hay chunks parecidos pero no responden: Gemini se abstiene y se conservan las citas."""
    frase = "No encuentro en los documentos cargados información que responda a esa pregunta."
    client = await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)
    client._respuesta = frase

    respuesta = await api.post("/query", json={"question": PREGUNTA, "top_k": 2})

    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["abstained"] is True
    assert cuerpo["abstain_reason"] == "modelo_abstuvo"
    assert cuerpo["answer"].strip() == frase
    assert cuerpo["citations"], "la abstención del modelo conserva la evidencia recuperada"
    assert client.prompt is not None


async def test_indice_vacio_da_503_empty_index(api, gemini):
    """Sin nada indexado no hay abstención: es un 503 para que la UI lo distinga."""
    respuesta = await api.post("/query", json={"question": PREGUNTA})

    assert respuesta.status_code == 503
    cuerpo = respuesta.json()
    assert cuerpo["code"] == "empty_index"
    assert "Ingesta documentos primero" in cuerpo["detail"]


# --- Citas (regla 03) ------------------------------------------------------------


async def test_las_citas_de_la_respuesta_mapean_a_citations(api, monkeypatch):
    """Cada `[n]` del texto tiene su chunk en `citations[index - 1]`."""
    client = await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)
    client.por_texto[PREGUNTA] = VEC_PREGUNTA
    client._respuesta = "Corta el tallo en 45 grados y retira el follaje bajo el agua [1]."

    respuesta = await api.post("/query", json={"question": PREGUNTA, "top_k": 3})

    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["abstained"] is False
    assert cuerpo["abstain_reason"] is None
    assert "[1]" in cuerpo["answer"]

    citas = cuerpo["citations"]
    assert citas, "una respuesta con citas debe traer sus citas"
    for cita in citas:
        assert cita["index"] >= 1
        assert cita["id"] == f"{cita['source']}#chunk-{cita['chunk_index']:03d}"
        assert cita["source"] == DOCUMENTO
        assert cita["text"].strip()
        assert 0.0 <= cita["score"] <= 1.0
    # El orden es por score descendente y `index` es correlativo: [1] es la mejor cita.
    assert [c["index"] for c in citas] == list(range(1, len(citas) + 1))
    scores = [c["score"] for c in citas]
    assert scores == sorted(scores, reverse=True)


async def test_el_prompt_numera_las_citas_como_las_devuelve_la_api(api, monkeypatch):
    """Los `[n]` del prompt son los mismos `index` que viaja en `citations` (regla 03)."""
    client = await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)
    client.por_texto[PREGUNTA] = VEC_PREGUNTA

    respuesta = await api.post("/query", json={"question": PREGUNTA, "top_k": 2})
    citas = respuesta.json()["citations"]

    for cita in citas:
        assert f"[{cita['index']}]" in client.prompt
    assert PREGUNTA in client.prompt
    assert "español" in client.prompt


async def test_el_filtro_por_fuente_solo_devuelve_citas_de_ese_documento(api, monkeypatch):
    await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)

    respuesta = await api.post("/query", json={"question": PREGUNTA, "source": DOCUMENTO})

    assert respuesta.status_code == 200, respuesta.text
    assert {c["source"] for c in respuesta.json()["citations"]} == {DOCUMENTO}


async def test_min_score_alto_abstiene_sin_generar(api, monkeypatch):
    """Con `min_score: 0.99` y score real ~0.95 no hay evidencia suficiente."""
    client = await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)
    client.por_texto[PREGUNTA] = [0.6, 0.8, 0.0, 0.0]  # coseno ~0.6 con el chunk

    respuesta = await api.post(
        "/query", json={"question": PREGUNTA, "min_score": 0.99}
    )

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["abstained"] is True
    assert cuerpo["abstain_reason"] == "sin_evidencia"
    assert client.prompt is None


class GeminiQueSeCuelga(GeminiFalso):
    """Reproduce el `ReadTimeout` real del SDK: agota `GOOGLE_TIMEOUT_SECONDS`."""

    def __init__(self, *, vector, al_vectorear: bool = False, al_generar: bool = False) -> None:
        super().__init__(vector=vector)
        self._al_vectorear = al_vectorear
        self._al_generar = al_generar

    def embed_content(self, *, model, contents, config=None):
        if self._al_vectorear:
            raise httpx.ReadTimeout("read operation timed out")
        return super().embed_content(model=model, contents=contents, config=config)

    def generate_content(self, *, model, contents, config=None):
        self.prompt = contents
        if self._al_generar:
            raise httpx.ReadTimeout("read operation timed out")
        return SimpleNamespace(text=self._respuesta)


async def test_timeout_de_embeddings_da_504(api, monkeypatch):
    """Google AI no responde a tiempo al vectorizar: 504, no 500 sin cuerpo."""
    await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)
    _instalar_gemini(monkeypatch, GeminiQueSeCuelga(vector=VEC_CHUNK, al_vectorear=True))

    respuesta = await api.post("/query", json={"question": PREGUNTA})

    assert respuesta.status_code == 504, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["code"] == "google_ai_timeout"
    assert "30 s" in cuerpo["detail"]


async def test_timeout_de_generacion_por_httpx_da_504(api, monkeypatch):
    """El timeout del SDK llega como excepción de `httpx`, no como `APIError`."""
    await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)
    _instalar_gemini(monkeypatch, GeminiQueSeCuelga(vector=VEC_CHUNK, al_generar=True))

    respuesta = await api.post("/query", json={"question": PREGUNTA})

    assert respuesta.status_code == 504, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["code"] == "google_ai_timeout"


# --- Contrato del endpoint -------------------------------------------------------


async def test_pregunta_vacia_da_400(api, gemini):
    respuesta = await api.post("/query", json={"question": "   "})

    assert respuesta.status_code == 400
    assert respuesta.json()["code"] == "empty_question"


async def test_sin_clave_da_503(api, gemini, monkeypatch):
    await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    respuesta = await api.post("/query", json={"question": PREGUNTA})

    assert respuesta.status_code == 503
    assert respuesta.json()["code"] == "missing_google_api_key"


async def test_fallo_de_gemini_da_503(api, monkeypatch):
    """Si Gemini revienta, es un error de infraestructura: nunca una abstención."""
    await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)

    _instalar_gemini(
        monkeypatch,
        GeminiQueFalla(
            vector=VEC_CHUNK,
            excepcion=genai_errors.APIError(code=500, response_json={"error": {"message": "boom"}}),
        ),
    )
    respuesta = await api.post("/query", json={"question": PREGUNTA})

    assert respuesta.status_code == 503
    assert respuesta.json()["code"] == "google_ai_error"


async def test_timeout_de_gemini_da_504(api, monkeypatch):
    await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)

    _instalar_gemini(
        monkeypatch,
        GeminiQueFalla(
            vector=VEC_CHUNK,
            excepcion=genai_errors.APIError(code=504, response_json={"error": {"message": "timeout"}}),
        ),
    )
    respuesta = await api.post("/query", json={"question": PREGUNTA})

    assert respuesta.status_code == 504
    assert respuesta.json()["code"] == "google_ai_timeout"


async def test_respuesta_vacia_de_gemini_da_503(api, monkeypatch):
    await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)

    _instalar_gemini(monkeypatch, GeminiQueFalla(vector=VEC_CHUNK, texto="   "))
    respuesta = await api.post("/query", json={"question": PREGUNTA})

    assert respuesta.status_code == 503
    assert respuesta.json()["code"] == "google_ai_error"


async def test_la_respuesta_incluye_modelos_y_tiempos(api, monkeypatch):
    from app.settings import get_settings

    await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)

    cuerpo = (await api.post("/query", json={"question": PREGUNTA, "top_k": 2})).json()
    ajustes = get_settings()

    assert cuerpo["question_embedding_model"] == ajustes.embedding_model
    assert cuerpo["generation_model"] == ajustes.generation_model
    assert cuerpo["top_k"] == 2
    assert cuerpo["retrieval_ms"] >= 0
    assert cuerpo["generation_ms"] >= 0


async def test_el_embedding_de_la_pregunta_usa_el_mismo_modelo_que_los_chunks(api, monkeypatch):
    """Regla 03: si se mezclan modelos, el k-NN no significa nada."""
    await _indexar_con_vector(api, monkeypatch, VEC_CHUNK)
    capturado: list[str] = []

    import app.embed as embed

    original = embed.GoogleAIEmbedder.embed_query

    async def espia(self, text):
        capturado.append(self.model)
        return list(VEC_PREGUNTA)

    monkeypatch.setattr(embed.GoogleAIEmbedder, "embed_query", espia)
    await api.post("/query", json={"question": PREGUNTA})

    from app.settings import get_settings

    assert capturado == [get_settings().embedding_model]