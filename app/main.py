"""FastAPI: orquesta ingesta, consulta y estado. Única capa que toca Chroma y Google AI."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.chunk import ChunkingError, chunk_text, es_formato_soportado, extract_text
from app.embed import EmbeddingError, GoogleAIEmbedder
from app.generate import FRASE_ABSTENCION, GenerationError, GoogleAIResponder
from app.models import (
    HealthStatus,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SkippedFile,
)
from app.settings import get_settings
from app.store import StoreError, add_chunks, count_chunks, ping, query_top_k

API_VERSION = "1.0.0"

settings = get_settings()

app = FastAPI(
    title="Sistema RAG — Catálogo de Flores y Listas de Precios",
    version=API_VERSION,
    description="API del sistema RAG: ingesta con embeddings de Google AI y consulta sobre ChromaDB.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ErrorApi(HTTPException):
    """Error con el cuerpo plano del esquema `Error`: `{detail, code, hint}`."""

    def __init__(self, status_code: int, detail: str, code: str, hint: str | None = None) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.code = code
        self.hint = hint


@app.exception_handler(ErrorApi)
async def _cuerpo_de_error(request: Request, exc: ErrorApi) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.code, "hint": exc.hint},
    )


@app.get(
    "/health",
    response_model=HealthStatus,
    tags=["salud"],
    operation_id="healthCheck",
    summary="Verifica el estado de la API, de ChromaDB y de la clave de Google AI.",
)
def health() -> HealthStatus:
    """Siempre 200: si Chroma no responde o falta la clave, `status: degraded`."""
    current = get_settings()
    chroma_accessible = ping(current)

    chunks = 0
    if chroma_accessible:
        from app.store import count_chunks

        try:
            chunks = count_chunks(current)
        except Exception:
            chroma_accessible = False

    key_configured = current.key_configured
    return HealthStatus(
        status="ok" if chroma_accessible and key_configured else "degraded",
        version=API_VERSION,
        chroma_accessible=chroma_accessible,
        google_ai_key_configured=key_configured,
        collection=current.chroma_collection,
        chunks_in_index=chunks,
        embedding_model=current.embedding_model,
        generation_model=current.generation_model,
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["ingesta"],
    operation_id="ingestDocuments",
    summary="Chunkifica, incrusta con Google AI y persiste en ChromaDB.",
)
async def ingest_documentos(requisito: IngestRequest = Body(...)) -> IngestResponse:
    """Trocea los documentos, vectoriza cada trozo con Google AI y lo guarda en Chroma.

    Siempre `200`: lo que no se puede indexar se devuelve en `skipped` con su motivo,
    porque la ingesta no es atómica y es preferible indexar el resto.
    """
    actual = get_settings()
    if not requisito.paths:
        raise ErrorApi(
            400,
            "No se recibió ningún archivo ni ruta. Usa `files` (subida) o `paths` (rutas del servidor).",
            "ingest_no_input",
        )
    if not actual.key_configured:
        raise ErrorApi(
            503,
            "GOOGLE_API_KEY no está configurada en el servidor.",
            "missing_google_api_key",
            "Copia .env.example a .env y define GOOGLE_API_KEY (https://aistudio.google.com/apikey).",
        )
    if not ping(actual):
        raise ErrorApi(503, f"ChromaDB no está accesible en {actual.chroma_path}.", "chroma_unavailable")
    if requisito.reset:
        _vaciar_indice(actual)

    embedder = GoogleAIEmbedder(actual)
    chunk_size = requisito.chunk_size or actual.chunk_size
    chunk_overlap = requisito.chunk_overlap or actual.chunk_overlap

    procesados: list[str] = []
    omitidos: list[SkippedFile] = []
    chunks_indexed = 0
    inicio = time.perf_counter()

    for ruta in _rutas_a_indexar(requisito.paths):
        source = _source_de(ruta)
        try:
            _comprobar_tamano(ruta, actual)
            trozos = chunk_text(
                extract_text(ruta),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source=source,
                title=Path(ruta).stem,
            )
        except ChunkingError as exc:
            omitidos.append(SkippedFile(source=source, reason=exc.code))
            continue

        try:
            vectores = await embedder.embed_documents([trozo.text for trozo in trozos])
            await add_chunks(actual, trozos, vectores)
        except (EmbeddingError, StoreError) as exc:
            omitidos.append(SkippedFile(source=source, reason=exc.code))
            continue

        procesados.append(source)
        chunks_indexed += len(trozos)

    return IngestResponse(
        documents_processed=len(procesados),
        chunks_indexed=chunks_indexed,
        chunks_skipped=0,
        collection=actual.chroma_collection,
        embedding_model=actual.embedding_model,
        sources=procesados,
        skipped=omitidos,
        duration_ms=int((time.perf_counter() - inicio) * 1000),
    )


# --- apoyo de la ingesta ---------------------------------------------------------


def _rutas_a_indexar(paths: list[str]) -> list[Path]:
    """Archivos a procesar: los indicados y, si son carpetas, los admitidos dentro.

    Una ruta explícita se conserva aunque no exista o no sea indexable, para que el
    error aparezca en `skipped`. Dentro de una carpeta solo se recorren los formatos
    admitidos: no es un error que un `data/` tenga un `.DS_Store`.
    """
    rutas: list[Path] = []
    for bruto in paths:
        ruta = Path(bruto).expanduser()
        if not ruta.is_absolute():
            ruta = Path.cwd() / ruta
        if ruta.is_dir():
            rutas.extend(
                hijo
                for hijo in sorted(ruta.rglob("*"))
                if hijo.is_file() and es_formato_soportado(hijo)
            )
        else:
            rutas.append(ruta)
    return rutas


def _source_de(ruta: Path) -> str:
    """`source` es la clave de filtro en Chroma: ruta relativa al proyecto si puede ser."""
    raiz = Path(__file__).resolve().parents[1]
    try:
        return ruta.resolve().relative_to(raiz).as_posix()
    except ValueError:
        return ruta.as_posix()


def _comprobar_tamano(ruta: Path, actual) -> None:
    if ruta.is_file() and ruta.stat().st_size > actual.max_file_bytes:
        raise ChunkingError(
            "demasiado_grande",
            f"El archivo `{ruta.name}` supera el límite de {actual.max_file_bytes} bytes.",
        )


def _vaciar_indice(actual) -> None:
    """`reset: true`: deja la colección vacía antes de indexar."""
    if count_chunks(actual) == 0:
        return
    from app.store import get_collection

    ids = get_collection(actual).get(include=[])["ids"] or []
    if ids:
        get_collection(actual).delete(ids=ids)

@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["consulta"],
    operation_id="queryRag",
    summary="Recupera top-k chunks en ChromaDB y genera una respuesta anclada con Gemini.",
)
async def consultar(requisito: QueryRequest = Body(...)) -> QueryResponse:
    """Pregunta al corpus. Una abstención es un `200` de negocio, nunca un `500`."""
    question = requisito.question.strip()
    if not question:
        raise ErrorApi(400, "La pregunta no puede estar vacía.", "empty_question")

    actual = get_settings()
    if not actual.key_configured:
        raise ErrorApi(
            503,
            "GOOGLE_API_KEY no está configurada en el servidor.",
            "missing_google_api_key",
            "Obtén la clave en https://aistudio.google.com/apikey",
        )
    if not ping(actual):
        raise ErrorApi(503, f"ChromaDB no está accesible en {actual.chroma_path}.", "chroma_unavailable")
    if count_chunks(actual) == 0:
        raise ErrorApi(
            503, "El índice está vacío. Ingesta documentos primero con POST /ingest.", "empty_index"
        )

    top_k = requisito.top_k or actual.top_k_default
    min_score = actual.min_score_default if requisito.min_score is None else requisito.min_score

    inicio = time.perf_counter()
    try:
        vector = await GoogleAIEmbedder(actual).embed_query(question)
        vecinos = await query_top_k(actual, vector, top_k, source=requisito.source)
    except EmbeddingError as exc:
        raise _error_de_embeddings(exc) from exc
    except StoreError as exc:
        raise ErrorApi(503, str(exc), "chroma_unavailable") from exc
    retrieval_ms = _milis_desde(inicio)

    # Capa determinista de la abstención: sin evidencia no se llama a Gemini.
    evidencia = [vecino for vecino in vecinos if vecino.score >= min_score]
    if not evidencia:
        return QueryResponse(
            answer=SIN_EVIDENCIA,
            abstained=True,
            abstain_reason="sin_evidencia",
            citations=[],
            question_embedding_model=actual.embedding_model,
            generation_model=None,
            top_k=top_k,
            retrieval_ms=retrieval_ms,
            generation_ms=0,
        )

    inicio = time.perf_counter()
    try:
        resultado = await GoogleAIResponder(actual).generate(question, evidencia)
    except GenerationError as exc:
        raise _error_de_generacion(exc) from exc

    return QueryResponse(
        answer=resultado.answer,
        abstained=resultado.abstained,
        abstain_reason="modelo_abstuvo" if resultado.abstained else None,
        citations=evidencia,
        question_embedding_model=actual.embedding_model,
        generation_model=actual.generation_model,
        top_k=top_k,
        retrieval_ms=retrieval_ms,
        generation_ms=_milis_desde(inicio),
    )


# --- apoyo de la consulta --------------------------------------------------------

SIN_EVIDENCIA = (
    "No tengo evidencia suficiente en el corpus indexado para responder a esa pregunta."
)


def _milis_desde(inicio: float) -> int:
    return int((time.perf_counter() - inicio) * 1000)


def _error_de_embeddings(exc: EmbeddingError) -> ErrorApi:
    if exc.code == "missing_google_api_key":
        return ErrorApi(503, str(exc), "missing_google_api_key")
    if exc.code == "google_ai_timeout":
        return ErrorApi(504, str(exc), "google_ai_timeout")
    return ErrorApi(
        503,
        f"Google AI falló al vectorizar la pregunta: {exc}",
        "google_ai_error",
        "Reintenta en unos segundos; si persiste, revisa la cuota en https://ai.dev/rate-limit.",
    )


def _error_de_generacion(exc: GenerationError) -> ErrorApi:
    if exc.code == "missing_google_api_key":
        return ErrorApi(503, str(exc), "missing_google_api_key")
    if exc.code == "google_ai_timeout":
        return ErrorApi(504, str(exc), "google_ai_timeout")
    return ErrorApi(
        503,
        f"Google AI falló al generar la respuesta: {exc}",
        "google_ai_error",
        "Reintenta en unos segundos; si persiste, revisa la cuota en https://ai.dev/rate-limit.",
    )
