"""Pruebas de `ui/streamlit_app.py`.

Dos niveles, sin abrir navegador ni tocar la red:

1. **Unidad** — se importa el módulo y se comprueban las funciones que hablan con la
   API, con `httpx.MockTransport` inyectado en `cliente`.
2. **Integración** — `streamlit.testing.v1.AppTest` ejecuta el script entero en
   proceso (como hace `streamlit run`) contra un **servidor HTTP real de mentira**
   levantado en un hilo. Así se prueba el render de verdad: respuestas, citas,
   abstenciones y errores.

Regla 06 verificada aquí: la UI no importa `app/`, ni `chromadb`, ni `google-genai`,
ni la clave de Google.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui" / "streamlit_app.py"

SALUD = {
    "status": "ok",
    "version": "1.0.0",
    "chroma_accessible": True,
    "google_ai_key_configured": True,
    "collection": "flowers_catalog",
    "chunks_in_index": 12,
    "embedding_model": "gemini-embedding-001",
    "generation_model": "gemini-3.5-flash",
}

CONSULTA_OK = {
    "answer": "Corta el tallo en diagonal a 45 grados [1] y retira el follaje bajo el agua [2].",
    "abstained": False,
    "abstain_reason": None,
    "citations": [
        {
            "index": 1,
            "id": "data/uploads/rosas.md#chunk-004",
            "source": "data/uploads/rosas.md",
            "title": "rosas",
            "chunk_index": 4,
            "page": None,
            "text": "Corte en diagonal a 45 grados para evitar que se obstruya el tallo.",
            "score": 0.83,
        },
        {
            "index": 2,
            "id": "data/uploads/rosas.md#chunk-009",
            "source": "data/uploads/rosas.md",
            "title": "rosas",
            "chunk_index": 9,
            "page": None,
            "text": "Retira el follaje que quede por debajo de la línea de agua.",
            "score": 0.61,
        },
    ],
    "question_embedding_model": "gemini-embedding-001",
    "generation_model": "gemini-3.5-flash",
    "top_k": 3,
    "retrieval_ms": 120,
    "generation_ms": 900,
}

ABSTENCION_SIN_EVIDENCIA = {
    "answer": "No tengo evidencia suficiente en el corpus indexado para responder a esa pregunta.",
    "abstained": True,
    "abstain_reason": "sin_evidencia",
    "citations": [],
    "question_embedding_model": "gemini-embedding-001",
    "generation_model": None,
    "top_k": 3,
    "retrieval_ms": 90,
    "generation_ms": 0,
}

ABSTENCION_DEL_MODELO = {
    "answer": "No encuentro en los documentos cargados información que responda a esa pregunta.",
    "abstained": True,
    "abstain_reason": "modelo_abstuvo",
    "citations": [
        {
            "index": 1,
            "id": "data/uploads/rosas.md#chunk-001",
            "source": "data/uploads/rosas.md",
            "title": "rosas",
            "chunk_index": 1,
            "page": None,
            "text": "Las rosas rosadas symbolism no aparece en ningún documento.",
            "score": 0.47,
        }
    ],
    "question_embedding_model": "gemini-embedding-001",
    "generation_model": "gemini-3.5-flash",
    "top_k": 3,
    "retrieval_ms": 110,
    "generation_ms": 1400,
}

INGESTA_OK = {
    "documents_processed": 1,
    "chunks_indexed": 3,
    "chunks_skipped": 0,
    "collection": "flowers_catalog",
    "embedding_model": "gemini-embedding-001",
    "sources": ["data/uploads/rosas.md"],
    "skipped": [],
    "duration_ms": 1500,
}


# --- servidor HTTP de mentira -----------------------------------------------------


class Servidor:
    """FastAPI de mentira en un hilo: registra lo que la UI le pide y responde lo fijo."""

    def __init__(self, respuestas: dict[tuple[str, str], object]) -> None:
        self.peticiones: list[tuple[str, str, object]] = []
        self.respuestas = respuestas
        self.servidor = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.hilo = threading.Thread(target=self.servidor.serve_forever, daemon=True)
        self.hilo.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.servidor.server_address[1]}"

    def parar(self) -> None:
        self.servidor.shutdown()
        self.servidor.server_close()

    def cuerpos(self, path: str) -> list[object]:
        return [cuerpo for _, p, cuerpo in self.peticiones if p == path]

    def _handler(self):
        guardar = self.peticiones
        respuestas = self.respuestas

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _responder(self, metodo: str) -> None:
                largo = int(self.headers.get("Content-Length") or 0)
                crudo = self.rfile.read(largo) if largo else b""
                cuerpo = json.loads(crudo) if crudo else None
                guardar.append((metodo, self.path.split("?")[0], cuerpo))
                clave = (metodo, self.path.split("?")[0])
                if clave in respuestas:
                    estado, carga = respuestas[clave]
                else:
                    estado, carga = 404, {"detail": "no mockeado", "code": "not_found"}
                datos = json.dumps(carga).encode()
                self.send_response(estado)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(datos)))
                self.end_headers()
                self.wfile.write(datos)

            def do_GET(self):
                self._responder("GET")

            def do_POST(self):
                self._responder("POST")

        return Handler


@pytest.fixture
def servidor():
    """Servidor stub con respuestas por defecto; los tests pueden sobrescribirlas."""
    por_defecto = {
        ("GET", "/health"): (200, SALUD),
        ("POST", "/ingest"): (200, INGESTA_OK),
        ("POST", "/query"): (200, CONSULTA_OK),
    }
    instancia = Servidor(por_defecto)
    yield instancia
    instancia.parar()


@pytest.fixture
def ui(servidor, tmp_path, monkeypatch):
    """Importa el script con `API_URL` apuntando al stub y sin tocar `data/uploads/`."""
    monkeypatch.setenv("API_URL", servidor.url)
    spec = importlib.util.spec_from_file_location("ui_streamlit_app", UI)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["ui_streamlit_app"] = modulo
    spec.loader.exec_module(modulo)
    modulo.CARPETA_SUBIDAS = tmp_path / "uploads"
    st_cache_resource_clear()
    return modulo


@pytest.fixture
def app_test(ui, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    # `AppTest` re-ejecuta el script, así que `UPLOADS_DIR` debe apuntar al temporal:
    # si no, la prueba escribiría en el `data/uploads/` real del repositorio.
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    # `st.cache_resource` es global: sin limpiarlo el cliente HTTP del test anterior
    # (apuntando a un servidor ya parado) se reutiliza y todo falla.
    st_cache_resource_clear()
    return AppTest.from_file(str(UI), default_timeout=30)


def st_cache_resource_clear() -> None:
    import streamlit as st

    st.cache_resource.clear()


def _markdown(app_test) -> str:
    return " ".join(m.value for m in app_test.markdown)


def _todo_el_texto(app_test) -> str:
    partes = [_markdown(app_test)]
    partes += [c.value for c in app_test.caption]
    partes += [e.value for e in app_test.error]
    partes += [w.value for w in app_test.warning]
    return " ".join(partes)


class Subido:
    """Doble de `UploadedFile` de Streamlit."""

    def __init__(self, nombre: str, contenido: bytes) -> None:
        self.name = nombre
        self._contenido = contenido

    def getvalue(self) -> bytes:
        return self._contenido


# --- aislamiento (regla 06) --------------------------------------------------------


def _imports_de_la_ui() -> set[str]:
    """Módulos que la UI importa de verdad (regla 06), leídos del AST."""
    arbol = ast.parse(UI.read_text(encoding="utf-8"))
    nombres: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            nombres.update(alias.name for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            nombres.add(nodo.module)
    return nombres


def test_la_ui_no_importa_app_ni_chromadb():
    """Regla 06: los imports reales de la UI, no las menciones del docstring."""
    imports = _imports_de_la_ui()

    assert not [nombre for nombre in imports if nombre == "app" or nombre.startswith("app.")], (
        "la UI no puede importar la capa de API"
    )
    assert "chromadb" not in imports
    assert not [nombre for nombre in imports if nombre.startswith("google.genai")]


def test_la_ui_solo_depende_de_httpx_y_streamlit():
    """Dependencias de la UI: las de apoyo, nada de la capa de índice o del modelo."""
    imports = _imports_de_la_ui()
    esperadas = {"httpx", "streamlit"}
    permitidos = esperadas | {"__future__", "os", "pathlib", "typing"}
    sobrantes = imports - permitidos
    assert not sobrantes, f"dependencias inesperadas en la UI: {sobrantes}"


def test_la_ui_no_maneja_la_clave_de_google():
    """La clave vive en el entorno del servidor: la UI no la lee ni la envía.

    Mencionar `missing_google_api_key` para traducir el error está bien; lo que no
    puede hacer la UI es *leer* la clave del entorno o mandarla en alguna petición.
    """
    source = UI.read_text(encoding="utf-8")
    assert 'os.environ["GOOGLE_API_KEY"]' not in source
    assert 'os.getenv("GOOGLE_API_KEY")' not in source
    assert "load_dotenv" not in source
    assert "Authorization" not in source
    assert "api_key=" not in source


# --- cliente HTTP ------------------------------------------------------------------


def test_consulta_el_health_para_el_banner(ui, servidor):
    salud = ui.estado()

    assert salud["status"] == "ok"
    assert ("GET", "/health", None) in servidor.peticiones


def test_pregunta_manda_question_top_k_y_min_score(ui, servidor):
    cuerpo = ui.preguntar("¿Cómo corto el tallo?", top_k=4, min_score=0.42)

    assert cuerpo["answer"].startswith("Corta el tallo")
    enviado = servidor.cuerpos("/query")[0]
    assert enviado["question"] == "¿Cómo corto el tallo?"
    assert enviado["top_k"] == 4
    assert enviado["min_score"] == 0.42


def test_pregunta_vacia_no_sale_a_la_red(ui, servidor):
    with pytest.raises(ValueError):
        ui.preguntar("   ", top_k=3, min_score=0.35)

    assert servidor.cuerpos("/query") == []


def test_el_filtro_por_documento_solo_se_manda_si_hay_source(ui, servidor):
    ui.preguntar("¿Y los tulipanes?", top_k=3, min_score=0.35, source="data/rosas.md")
    ui.preguntar("¿Y los tulipanes?", top_k=3, min_score=0.35)

    enviados = servidor.cuerpos("/query")
    assert enviados[0]["source"] == "data/rosas.md"
    assert "source" not in enviados[1]


def test_indexar_guarda_el_archivo_y_manda_su_ruta(ui, servidor, tmp_path):
    resultado = ui.indexar_archivos(
        [Subido("rosas.md", b"# Rosas\nCuanto mas solo mejor.")],
        chunk_size=320,
        chunk_overlap=64,
        reset=False,
    )

    assert resultado["chunks_indexed"] == 3
    assert (tmp_path / "uploads" / "rosas.md").read_bytes() == b"# Rosas\nCuanto mas solo mejor."
    enviado = servidor.cuerpos("/ingest")[0]
    assert enviado["paths"] == [str(tmp_path / "uploads" / "rosas.md")]
    assert enviado["chunk_size"] == 320
    assert enviado["chunk_overlap"] == 64
    assert enviado["reset"] is False


def test_indexar_sin_archivos_no_llama_a_la_api(ui, servidor):
    with pytest.raises(ValueError):
        ui.indexar_archivos([], chunk_size=320, chunk_overlap=64, reset=False)

    assert servidor.cuerpos("/ingest") == []


def test_error_de_la_api_propaga_code_detalle_y_hint(servidor, monkeypatch, ui):
    servidor.respuestas[("POST", "/query")] = (
        503,
        {
            "detail": "El índice está vacío. Ingesta documentos primero con POST /ingest.",
            "code": "empty_index",
            "hint": "Sube un documento y pulsa Indexar.",
        },
    )

    with pytest.raises(ui.ErrorAPI) as exc:
        ui.preguntar("¿Cuánto cuesta?", top_k=3, min_score=0.35)

    assert exc.value.code == "empty_index"
    assert exc.value.status == 503
    assert "Ingesta documentos" in exc.value.detail
    assert ui.aviso(exc.value).startswith("El índice está vacío")


def test_api_caida_da_un_error_legible_no_una_excepcion_de_httpx(monkeypatch, ui, tmp_path):
    monkeypatch.setattr(ui, "cliente", lambda: httpx.Client(base_url="http://127.0.0.1:9", timeout=0.2))

    with pytest.raises(ui.ErrorAPI) as exc:
        ui.estado()

    assert exc.value.code == "api_unreachable"
    assert "uvicorn app.main:app" in ui.aviso(exc.value)


# --- render del chat (AppTest) ----------------------------------------------------


def test_la_app_arranca_sin_excepcion(app_test):
    app_test.run()

    assert not app_test.exception
    assert "Catálogo de Flores" in _markdown(app_test)


def test_el_banner_muestra_chunks_y_modelos(app_test):
    app_test.run()

    texto = _todo_el_texto(app_test)
    assert "12 chunks" in texto
    assert "gemini-embedding-001" in texto
    assert "gemini-3.5-flash" in texto


def test_la_pregunta_se_muestra_como_turno_de_usuario(app_test):
    app_test.run()
    app_test.chat_input[0].set_value("¿Cómo corto el tallo?").run()

    assert not app_test.exception
    assert "¿Cómo corto el tallo?" in _markdown(app_test)


def test_la_respuesta_de_gemini_se_pinta(app_test):
    app_test.run()
    app_test.chat_input[0].set_value("¿Cómo corto el tallo?").run()

    assert "Corta el tallo en diagonal a 45 grados [1]" in _markdown(app_test)


def test_las_citas_se_muestran_debajo_con_texto_origen_y_score(app_test):
    """Regla 03: por cada `[n]` hay una cita con su `source`, su `score` y su `text`."""
    app_test.run()
    app_test.chat_input[0].set_value("¿Cómo corto el tallo?").run()

    assert not app_test.exception
    texto = _markdown(app_test)
    assert "Corte en diagonal a 45 grados para evitar que se obstruya el tallo." in texto
    assert "data/uploads/rosas.md" in texto
    assert "score 0.83" in texto
    assert "score 0.61" in texto
    assert "Fuentes (2)" in [e.label for e in app_test.expander]


def test_la_abstencion_sin_evidencia_no_se_muestra_como_error(app_test, servidor):
    servidor.respuestas[("POST", "/query")] = (200, ABSTENCION_SIN_EVIDENCIA)
    app_test.run()
    app_test.chat_input[0].set_value("¿Quién ganó el Mundial de 1998?").run()

    assert not app_test.exception
    assert not app_test.error, "una abstención es un 200 de negocio, no un error"
    texto = _markdown(app_test)
    assert "Sin evidencia suficiente" in texto
    assert "evidencia suficiente" in texto


def test_la_abstencion_del_modelo_conserva_las_citas(app_test, servidor):
    servidor.respuestas[("POST", "/query")] = (200, ABSTENCION_DEL_MODELO)
    app_test.run()
    app_test.chat_input[0].set_value("¿Qué color de rosa es el más vendido?").run()

    assert not app_test.error
    texto = _markdown(app_test)
    assert "El modelo se abstuvo" in texto
    assert "score 0.47" in texto


def test_el_error_de_la_api_se_muestra_con_su_consejo(app_test, servidor):
    servidor.respuestas[("POST", "/query")] = (
        503,
        {
            "detail": "El índice está vacío.",
            "code": "empty_index",
            "hint": None,
        },
    )
    app_test.run()
    app_test.chat_input[0].set_value("¿Cuánto cuesta un ramo?").run()

    assert not app_test.exception
    assert any("El índice está vacío" in e.value for e in app_test.error)


def test_el_historial_se_conserva_entre_preguntas(app_test):
    app_test.run()
    app_test.chat_input[0].set_value("Primera pregunta").run()
    app_test.chat_input[0].set_value("Segunda pregunta").run()

    assert not app_test.exception
    mensajes = app_test.session_state["mensajes"]
    assert [m["rol"] for m in mensajes] == ["usuario", "asistente", "usuario", "asistente"]
    markdown = _markdown(app_test)
    assert "Primera pregunta" in markdown
    assert "Segunda pregunta" in markdown


def test_la_barra_lateral_tiene_subida_de_archivos_y_boton_indexar(app_test):
    app_test.run()

    assert len(app_test.file_uploader) == 1
    subida = app_test.file_uploader[0]
    assert set(subida.allowed_type) == {".md", ".txt", ".pdf", ".csv"}
    assert subida.accept_multiple_files is True
    assert any(b.label == "Indexar" for b in app_test.button)


def test_indexar_desde_la_ui_sube_el_archivo_y_avisa_los_omitidos(app_test, servidor, ui):
    """Flujo completo de la barra lateral: subir → Indexar → resumen + omitidos."""
    servidor.respuestas[("POST", "/ingest")] = (
        200,
        {
            **INGESTA_OK,
            "documents_processed": 1,
            "chunks_indexed": 5,
            "skipped": [{"source": "data/uploads/escan.pdf", "reason": "sin_capa_de_texto"}],
        },
    )
    app_test.run()
    app_test.file_uploader[0].set_value(("rosas.md", b"# Rosas\nCuanto mas solo mejor.", "text/markdown"))

    indice = [i for i, b in enumerate(app_test.button) if b.label == "Indexar"][0]
    app_test.button[indice].click().run()

    assert not app_test.exception, [e.value for e in app_test.error]
    assert servidor.cuerpos("/ingest"), "la UI tuvo que llamar a /ingest"
    enviado = servidor.cuerpos("/ingest")[0]
    assert enviado["paths"][0].endswith("uploads/rosas.md")
    assert any("Indexados 5 chunks" in s.value for s in app_test.success)
    assert any("escan.pdf" in w.value and "sin_capa_de_texto" in w.value for w in app_test.warning)


def test_indexar_sin_archivos_avisa_y_no_llama_a_la_api(app_test, servidor):
    app_test.run()
    indice = [i for i, b in enumerate(app_test.button) if b.label == "Indexar"][0]
    app_test.button[indice].click().run()

    assert not app_test.exception
    assert servidor.cuerpos("/ingest") == []
    assert any("archivos" in w.value.lower() for w in app_test.warning)
