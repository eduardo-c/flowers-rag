"""UI de Streamlit: cliente HTTP puro de la API de FastAPI.

Regla 06: este archivo **no importa nada de `app/`**, ni `chromadb`, ni
`google-genai`. Toda la inteligencia (chunking, embeddings, k-NN, Gemini) vive en
FastAPI; la UI solo muestra lo que la API devuelve.

Dos piezas:

* **Barra lateral** — `st.file_uploader` + botón *Indexar*. Cada archivo se escribe
  en `data/uploads/` y se llama a `POST /ingest` (modo JSON, con `paths`). Así la UI
  no depende de `multipart/form-data` y el servidor lee de su propio disco.
* **Chat principal** — `st.chat_input` → `POST /query`. Cada respuesta de Gemini se
  pinta con sus citas `[n]` debajo, con `source`, `score` y el texto del chunk, para
  que se pueda comprobar que la respuesta está anclada en la evidencia.

Ejecución: `streamlit run ui/streamlit_app.py --server.port 8501`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Protocol

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
# `UPLOADS_DIR` permite aislar la carpeta en pruebas y despliegues; por defecto `data/uploads`.
CARPETA_SUBIDAS = Path(
    os.getenv("UPLOADS_DIR") or Path(__file__).resolve().parents[1] / "data" / "uploads"
)
FORMATOS = ("md", "txt", "pdf", "csv")
TIMEOUT_S = 120.0

TITULO = "Catálogo de Flores · RAG"
SUBTITULO = "Pregunta sobre tus documentos. Las respuestas solo usan lo que está indexado."


class ArchivoSubido(Protocol):
    """Lo mínimo de `st.runtime.uploaded_file_manager.UploadedFile` que usa la UI."""

    name: str

    def getvalue(self) -> bytes: ...


class ErrorAPI(Exception):
    """Error de la API con su `code` estable, para que la UI elija el mensaje."""

    def __init__(self, status: int, detail: str, code: str, hint: str | None = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.code = code
        self.hint = hint


# --- cliente HTTP ------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def cliente() -> httpx.Client:
    """Una conexión por sesión de Streamlit, reutilizada en cada rerun."""
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT_S)


def _get(ruta: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return _cuerpo(_pedir("GET", ruta, params=params))


def _post(ruta: str, cuerpo: dict[str, Any]) -> dict[str, Any]:
    return _cuerpo(_pedir("POST", ruta, json=cuerpo))


def _pedir(metodo: str, ruta: str, **kwargs: Any) -> httpx.Response:
    try:
        return cliente().request(metodo, ruta, **kwargs)
    except httpx.HTTPError as exc:
        raise ErrorAPI(
            0,
            f"No se pudo conectar con la API en {API_URL}: {exc}",
            "api_unreachable",
            "Levanta FastAPI con `uvicorn app.main:app --port 8000`.",
        ) from exc


def _cuerpo(respuesta: httpx.Response) -> dict[str, Any]:
    if respuesta.status_code >= 400:
        try:
            error = respuesta.json()
        except ValueError:
            error = {}
        raise ErrorAPI(
            respuesta.status_code,
            error.get("detail") or respuesta.text or "Error sin detalle.",
            error.get("code", "unknown_error"),
            error.get("hint"),
        )
    return respuesta.json()


# --- operaciones -------------------------------------------------------------------


def estado() -> dict[str, Any]:
    """`GET /health` para el banner de estado."""
    return _get("/health")


def indexar_archivos(
    archivos: Iterable[ArchivoSubido],
    *,
    chunk_size: int,
    chunk_overlap: int,
    reset: bool,
) -> dict[str, Any]:
    """Escribe las subidas en `data/uploads/` y llama a `POST /ingest` con sus rutas."""
    rutas = _guardar(archivos)
    if not rutas:
        raise ValueError("No hay archivos que indexar.")
    return _post(
        "/ingest",
        {
            "paths": [str(ruta) for ruta in rutas],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "reset": reset,
        },
    )


def preguntar(
    question: str,
    *,
    top_k: int = 3,
    min_score: float = 0.35,
    source: str | None = None,
) -> dict[str, Any]:
    """`POST /query`. Una pregunta vacía no sale a la red: la UI la filtra antes."""
    if not question or not question.strip():
        raise ValueError("Escribe una pregunta antes de enviarla.")
    cuerpo: dict[str, Any] = {
        "question": question.strip(),
        "top_k": top_k,
        "min_score": min_score,
    }
    if source:
        cuerpo["source"] = source
    return _post("/query", cuerpo)


def _guardar(archivos: Iterable[ArchivoSubido]) -> list[Path]:
    CARPETA_SUBIDAS.mkdir(parents=True, exist_ok=True)
    rutas: list[Path] = []
    for archivo in archivos:
        destino = CARPETA_SUBIDAS / Path(archivo.name).name
        destino.write_bytes(archivo.getvalue())
        rutas.append(destino)
    return rutas


# --- errores -----------------------------------------------------------------------


AVISOS = {
    "api_unreachable": "La API no responde. Levanta FastAPI con `uvicorn app.main:app --port 8000`.",
    "missing_google_api_key": "La API no tiene GOOGLE_API_KEY: copia .env.example a .env, define la clave y reinicia FastAPI.",
    "empty_index": "El índice está vacío: sube documentos y pulsa Indexar antes de preguntar.",
    "chroma_unavailable": "La API no ve ChromaDB. Revisa CHROMA_PATH y reinicia FastAPI.",
    "google_ai_timeout": "Google AI tardó demasiado. Espera unos segundos y vuelve a intentarlo.",
    "google_ai_error": "Google AI devolvió un error (cuota o modelo). Revisa los modelos y la cuota de la clave.",
    "file_too_large": "Algún archivo supera el tamaño máximo permitido por la API.",
    "ingest_no_input": "No se recibió ningún archivo legible.",
}


def aviso(exc: ErrorAPI) -> str:
    """Mensaje para la persona usuaria: el consejo del código, más el detalle de la API."""
    return AVISOS.get(exc.code, exc.detail)


# --- render ------------------------------------------------------------------------

ESTILO = """
<style>
  .bloque-app { max-width: 62rem; padding-top: 1.5rem; }
  .titulo-app { font-weight: 700; letter-spacing: -0.02em; margin-bottom: .1rem; }
  .subtitulo-app { color: #6b7280; margin-bottom: 1.2rem; font-size: .95rem; }
  .pastilla { display:inline-flex; align-items:center; gap:.4rem; padding:.15rem .55rem;
              border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.02em;
              border:1px solid transparent; margin-right:.3rem; }
  .pastilla-ok { background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }
  .pastilla-aviso { background:#fffbeb; color:#92400e; border-color:#fde68a; }
  .pastilla-error { background:#fef2f2; color:#991b1b; border-color:#fecaca; }
  .cita { border:1px solid #e5e7eb; border-left:3px solid #9ca3af; border-radius:.5rem;
          padding:.6rem .85rem; margin:.45rem 0; background:#fafafa; }
  .cita-cabecera { display:flex; gap:.6rem; align-items:center; font-size:.78rem;
                   color:#4b5563; margin-bottom:.3rem; }
  .cita-texto { font-size:.9rem; color:#1f2937; }
  .fuente { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.75rem; }
</style>
"""


def _escapar(texto: Any) -> str:
    return str(texto).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pastilla(texto: str, clase: str = "aviso") -> str:
    return f'<span class="pastilla pastilla-{clase}">{_escapar(texto)}</span>'


def _cabecera() -> None:
    st.markdown(
        f'<div class="bloque-app"><div class="titulo-app">{_escapar(TITULO)}</div>'
        f'<div class="subtitulo-app">{_escapar(SUBTITULO)}</div></div>',
        unsafe_allow_html=True,
    )


def _resumen_indice(salud: dict[str, Any] | None) -> None:
    if not salud:
        return
    st.sidebar.markdown(
        _pastilla("API en línea", "ok") + _pastilla(f"{salud.get('chunks_in_index', 0)} chunks"),
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        f"Embeddings: `{salud.get('embedding_model', '?')}`  \n"
        f"Generación: `{salud.get('generation_model') or '—'}`"
    )


def pintar_citas(citas: list[dict[str, Any]]) -> None:
    """Las citas van debajo de la respuesta: `[n]`, origen, score y texto del chunk."""
    if not citas:
        return
    with st.expander(f"Fuentes ({len(citas)})"):
        for cita in citas:
            indice = cita.get("index")
            score = float(cita.get("score") or 0.0)
            origen = cita.get("source") or cita.get("id") or "desconocido"
            pagina = cita.get("page")
            donde = f"{origen} · pág. {pagina}" if pagina else str(origen)
            st.markdown(
                f'<div class="cita">'
                f'<div class="cita-cabecera">'
                f"{_pastilla(f'[{indice}]')} "
                f'<span class="fuente">{_escapar(donde)}</span>'
                f"<span>score {score:.2f}</span></div>"
                f'<div class="cita-texto">{_escapar(cita.get("text", ""))}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _pie(cuerpo: dict[str, Any]) -> None:
    st.caption(
        f"recuperación {cuerpo.get('retrieval_ms', 0)} ms · "
        f"generación {cuerpo.get('generation_ms', 0)} ms · "
        f"top_k {cuerpo.get('top_k', 0)}"
    )


def pintar_respuesta(cuerpo: dict[str, Any]) -> None:
    """Respuesta de Gemini + citas. Una abstención se marca, no parece un error."""
    if cuerpo.get("abstained"):
        motivo = cuerpo.get("abstain_reason") or "sin_evidencia"
        etiqueta = {
            "sin_evidencia": "Sin evidencia suficiente",
            "modelo_abstuvo": "El modelo se abstuvo",
        }.get(motivo, "Abstención")
        st.markdown(_pastilla(etiqueta), unsafe_allow_html=True)
    st.markdown(cuerpo.get("answer", ""))
    pintar_citas(cuerpo.get("citations") or [])
    _pie(cuerpo)


def pintar_bienvenida() -> None:
    with st.chat_message("assistant", avatar="🌹"):
        st.markdown(
            "Sube un documento en la barra lateral y pulsa **Indexar**. Después pregunta "
            "lo que quieras: solo responderé con lo que esté en el índice, y si no hay "
            "evidencia te lo diré en lugar de inventar."
        )


def _barra_lateral() -> tuple[list[Any], int, int, bool, bool]:
    with st.sidebar:
        st.header("Documentos")
        subida = st.file_uploader(
            "Sube .md, .txt, .pdf o .csv",
            type=list(FORMATOS),
            accept_multiple_files=True,
        )
        with st.expander("Parámetros de troceado"):
            chunk_size = st.slider("Palabras por chunk", 100, 1000, 320, step=20)
            chunk_overlap = st.slider("Solape entre chunks", 0, 200, 64, step=4)
            reset = st.checkbox("Vaciar el índice antes de indexar", value=False)
        indexar = st.button("Indexar", type="primary", use_container_width=True)
    return list(subida or []), chunk_size, chunk_overlap, reset, indexar


def _conectar() -> tuple[dict[str, Any] | None, ErrorAPI | None]:
    try:
        return estado(), None
    except ErrorAPI as exc:
        return None, exc


def main() -> None:
    st.set_page_config(page_title=TITULO, page_icon="🌹", layout="centered")
    st.markdown(ESTILO, unsafe_allow_html=True)

    st.session_state.setdefault("mensajes", [])
    subida, chunk_size, chunk_overlap, reset, pulsar_indexar = _barra_lateral()

    salud, error_salud = _conectar()
    if error_salud is not None:
        st.sidebar.error(aviso(error_salud))
    else:
        _resumen_indice(salud)

    if pulsar_indexar:
        with st.spinner("Trocendo, incrustando con Google AI y guardando en ChromaDB…"):
            try:
                resultado = indexar_archivos(
                    subida, chunk_size=chunk_size, chunk_overlap=chunk_overlap, reset=reset
                )
            except ErrorAPI as exc:
                st.error(aviso(exc))
            except ValueError as exc:
                st.warning(str(exc))
            else:
                st.success(
                    f"Indexados {resultado.get('chunks_indexed', 0)} chunks de "
                    f"{resultado.get('documents_processed', 0)} documentos."
                )
                for item in resultado.get("skipped") or []:
                    st.warning(f"Omitido `{item.get('source')}`: {item.get('reason')}")

    _cabecera()
    if not st.session_state.mensajes:
        pintar_bienvenida()

    top_k = st.slider("Chunks a recuperar (top_k)", 1, 10, 3)
    min_score = st.slider("Score mínimo para responder", 0.0, 1.0, 0.35, step=0.05)

    for mensaje in st.session_state.mensajes:
        _pintar_mensaje(mensaje)

    pregunta = st.chat_input("Escribe tu pregunta sobre los documentos indexados…")
    if not pregunta:
        return

    st.session_state.mensajes.append({"rol": "usuario", "texto": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar="🌹"):
        with st.spinner("Recuperando evidencia y redactando…"):
            cuerpo: dict[str, Any] | None = None
            fallo: ErrorAPI | None = None
            try:
                cuerpo = preguntar(pregunta, top_k=top_k, min_score=min_score)
            except ErrorAPI as exc:
                fallo = exc
            if fallo is not None:
                st.error(aviso(fallo))
            elif cuerpo is not None:
                pintar_respuesta(cuerpo)
                st.session_state.mensajes.append({"rol": "asistente", "respuesta": cuerpo})


def _pintar_mensaje(mensaje: dict[str, Any]) -> None:
    """Repinta un turno ya resuelto del historial."""
    if mensaje["rol"] == "usuario":
        with st.chat_message("user"):
            st.markdown(mensaje["texto"])
        return
    with st.chat_message("assistant", avatar="🌹"):
        pintar_respuesta(mensaje["respuesta"])


if __name__ == "__main__":
    main()
