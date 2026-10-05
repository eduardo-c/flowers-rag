"""Lectura de documentos y troceado en palabras con solape.

Unidad: **palabras**, no caracteres, porque los tokens del modelo cambian según el
idioma y la versión. El solape existe para que un dato que cae justo en el borde
entre dos trozos siga completo en alguno de los dos: con `stride = chunk_size -
chunk_overlap` las últimas `chunk_overlap` palabras de un trozo son las primeras
del siguiente, y al concatenar todos los trozos no se pierde ninguna palabra.

Aquí no se llama a Google AI ni a ChromaDB: solo se leen archivos y se cortan.
Contrato en `x-modulos` → `app/chunk.py` de `docs/openapi_spec.yaml`.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.models import Chunk

FORMATO_SOPORTADO = {".md", ".txt", ".csv", ".pdf"}


class ChunkingError(RuntimeError):
    """Falla al leer o trocear. `code` es estable (ver `x-modulos` del spec)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def es_formato_soportado(path: str | Path) -> bool:
    """`True` si la extensión está entre las admitidas por `/ingest`."""
    return Path(path).suffix.lower() in FORMATO_SOPORTADO


def extract_text(path: str | Path) -> str:
    """Texto plano de un `.md`, `.txt`, `.csv` o `.pdf` (con `pypdf`)."""
    ruta = Path(path)

    if not es_formato_soportado(ruta):
        raise ChunkingError(
            "formato_no_soportado",
            f"`{ruta.name}` no es un formato admitido: "
            f"{', '.join(sorted(FORMATO_SOPORTADO))}.",
        )
    if not ruta.is_file():
        raise ChunkingError("ilegible", f"No existe el archivo `{ruta}`.")

    if ruta.suffix.lower() == ".pdf":
        texto = _texto_pdf(ruta)
    else:
        try:
            texto = ruta.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ChunkingError("ilegible", f"No se pudo leer `{ruta}`: {exc}") from exc

    if not texto.strip():
        raise ChunkingError("vacio", f"`{ruta.name}` no tiene texto para indexar.")
    return texto


def chunk_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    source: str,
    title: str | None = None,
    page: int | None = None,
) -> list[Chunk]:
    """Trocea `text` en trozos de `chunk_size` palabras con `chunk_overlap` de solape.

    Devuelve un `Chunk` por trozo con `chunk_index` desde 0 y `source` como clave de
    filtro en ChromaDB.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size debe ser >= 1.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"chunk_overlap debe estar entre 0 y chunk_size - 1 "
            f"(0 <= {chunk_overlap} < {chunk_size})."
        )

    palabras = re.sub(r"\s+", " ", text).strip().split()
    if not palabras:
        return []

    stride = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    for indice, inicio in enumerate(range(0, len(palabras), stride)):
        trozo = palabras[inicio : inicio + chunk_size]
        if not trozo:
            break
        chunks.append(
            Chunk(
                source=source,
                text=" ".join(trozo),
                chunk_index=indice,
                title=title,
                page=page,
            )
        )
        if inicio + chunk_size >= len(palabras):
            break
    return chunks


def _texto_pdf(ruta: Path) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(ruta))
        paginas = [(pagina.extract_text() or "") for pagina in reader.pages]
    except Exception as exc:
        raise ChunkingError("ilegible", f"No se pudo leer el PDF `{ruta.name}`: {exc}") from exc

    texto = "\n\n".join(paginas).strip()
    if not texto:
        raise ChunkingError(
            "sin_capa_de_texto",
            f"`{ruta.name}` no tiene capa de texto (parece escaneado): hace falta OCR.",
        )
    return texto