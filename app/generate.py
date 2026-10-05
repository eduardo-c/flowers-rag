"""Generación de la respuesta con Gemini: español, solo evidencia y citas `[n]`.

Regla 03: el texto se redacta **exclusivamente** con los chunks recuperados y cada
afirmación lleva su `[n]`, que es el `index` 1-based del `RetrievedChunk` y por tanto
la posición en `citations[index - 1]` de la respuesta de la API.

La **capa determinista** de la abstención (filtrar por `min_score` y no llamar a
Gemini) vive en `app/main.py`: aquí no se decide si hay evidencia, solo se redacta
con la que le pasan. Si el modelo tampoco encuentra respuesta, devuelve la frase de
abstención acordada y `GenerationResult.abstained` lo refleja.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.models import RetrievedChunk
from app.settings import Settings, get_settings

FRASE_ABSTENCION = (
    "No encuentro en los documentos cargados información que responda a esa pregunta."
)

# El SDK lanza la excepción de `httpx` al agotarse `GOOGLE_TIMEOUT_SECONDS`; no es un
# `APIError`, así que sin esto se escaparía como un 500 sin cuerpo.
TIEMPO_ESPERADO = (TimeoutError, httpx.TimeoutException)

PROMPT = """Eres un asistente que responde EXCLUSIVAMENTE con la evidencia proporcionada.
Reglas:
1. Usa solo los fragmentos [1]..[n]. No uses conocimiento propio.
2. Cita cada afirmación con [n] usando exactamente esos números.
3. Responde siempre en español.
4. Si la evidencia no responde a la pregunta, responde exactamente
"{frase}" y nada más."""


@dataclass(frozen=True)
class GenerationResult:
    """Respuesta de Gemini y si se abstuvo por falta de evidencia."""

    answer: str
    abstained: bool


class GenerationError(RuntimeError):
    """Falla al generar. `code` es estable (ver `x-modulos` del spec)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class GoogleAIResponder:
    """Cliente de generación de Google AI Studio.

    `client` solo se inyecta en pruebas; en producción se construye con
    `GOOGLE_API_KEY` y el timeout de `GOOGLE_TIMEOUT_SECONDS`.
    """

    def __init__(self, settings: Settings | None = None, client: object | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = client

    @property
    def model(self) -> str:
        return self._settings.generation_model

    async def generate(
        self, question: str, citations: Sequence[RetrievedChunk]
    ) -> GenerationResult:
        """Redacta la respuesta en español con esos chunks y sus números `[n]`."""
        if not question or not question.strip():
            raise ValueError("La pregunta no puede estar vacía.")
        if not citations:
            raise ValueError(
                "Sin evidencia no se genera nada: el endpoint debe abstenerse antes de llamar."
            )

        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model, contents=self._prompt(question, citations)
            )
        except genai_errors.APIError as exc:
            raise self._error(exc) from exc
        except TIEMPO_ESPERADO as exc:
            raise GenerationError(
                "google_ai_timeout",
                f"Google AI tardó más de {self._settings.google_timeout_seconds} s en responder.",
            ) from exc

        answer = (getattr(response, "text", "") or "").strip()
        if not answer:
            raise GenerationError("invalid_response", "Gemini devolvió una respuesta vacía.")

        return GenerationResult(answer=answer, abstained=self._es_abstencion(answer))

    # --- interno ---------------------------------------------------------------

    def _prompt(self, question: str, citations: Sequence[RetrievedChunk]) -> str:
        """Reglas estrictas + evidencia numerada + pregunta."""
        evidencia = "\n\n".join(
            f"[{cita.index}] Fuente: {cita.source}\n{cita.text}" for cita in citations
        )
        reglas = PROMPT.format(frase=FRASE_ABSTENCION)
        return f"{reglas}\n\nEVIDENCIA:\n{evidencia}\n\nPREGUNTA: {question}"

    def _es_abstencion(self, answer: str) -> bool:
        """Gemini se abstiene si su respuesta es (o empieza por) la frase acordada."""
        return FRASE_ABSTENCION in answer

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client

        key = self._settings.google_api_key
        if not key:
            raise GenerationError(
                "missing_google_api_key", "GOOGLE_API_KEY no está configurada en el servidor."
            )
        timeout_ms = self._settings.google_timeout_seconds * 1000
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        return self._client

    def _error(self, exc: genai_errors.APIError) -> GenerationError:
        codigo = getattr(exc, "code", None)
        if codigo in (408, 504):
            return GenerationError(
                "google_ai_timeout",
                f"Google AI tardó más de {self._settings.google_timeout_seconds} s en responder.",
            )
        return GenerationError("google_ai_error", f"Google AI falló al generar con {self.model}: {exc}")