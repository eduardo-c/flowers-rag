"""Embeddings: texto -> vector, **solo** con Google AI (`google-genai`).

Único motor de embeddings del proyecto. Prohibido cualquier backend local
(FastText, BERT local, sentence-transformers, fastembed, bag-of-words) y
prohibido el embedder por defecto de ChromaDB: aquí no se importa `chromadb`.

El mismo modelo (`EMBEDDING_MODEL`) vectoriza los documentos y las preguntas;
si se mezclaran modelos distintos, el k-NN no significaría nada.
"""

from __future__ import annotations

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.settings import Settings, get_settings


class EmbeddingError(RuntimeError):
    """Falla al vectorizar. `code` es estable (ver `x-modulos` del spec)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class GoogleAIEmbedder:
    """Cliente de embeddings de Google AI Studio.

    `client` solo se inyecta en pruebas; en producción se construye con
    `GOOGLE_API_KEY` y el timeout de `GOOGLE_TIMEOUT_SECONDS`.
    """

    def __init__(self, settings: Settings | None = None, client: object | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._settings.embedding_model

    @property
    def dimension(self) -> int | None:
        """Dimensión del último vector recibido (`None` si aún no se vectorizó nada)."""
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectoriza documentos en lotes de `EMBED_BATCH_SIZE` (respetando la cuota)."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for inicio in range(0, len(texts), self._settings.embed_batch_size):
            lote = texts[inicio : inicio + self._settings.embed_batch_size]
            vectors.extend(self._embed(lote))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Vectoriza la pregunta con el mismo modelo que los documentos."""
        if not text or not text.strip():
            raise ValueError("La pregunta no puede estar vacía.")
        return self._embed([text])[0]

    # --- interno ---------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        try:
            response = client.models.embed_content(model=self.model, contents=list(texts))
        except genai_errors.APIError as exc:
            raise EmbeddingError(
                "google_ai_error",
                f"Google AI falló al vectorizar con {self.model}: {exc}",
            ) from exc

        values = [list(item.values) for item in response.embeddings]
        if len(values) != len(texts):
            raise EmbeddingError(
                "invalid_response",
                f"Google AI devolvió {len(values)} vectores para {len(texts)} textos.",
            )
        return [self._validated(vector) for vector in values]

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client

        key = self._settings.google_api_key
        if not key:
            raise EmbeddingError(
                "missing_google_api_key",
                "GOOGLE_API_KEY no está configurada en el servidor.",
            )
        timeout_ms = self._settings.google_timeout_seconds * 1000
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        return self._client

    def _validated(self, vector: list[float]) -> list[float]:
        if not vector:
            raise EmbeddingError("invalid_response", "Google AI devolvió un vector vacío.")
        if self._dimension is None:
            self._dimension = len(vector)
        elif len(vector) != self._dimension:
            raise EmbeddingError(
                "invalid_response",
                f"Dimensión cambiada de {self._dimension} a {len(vector)}: "
                "el índice habría que reindexarlo.",
            )
        return vector