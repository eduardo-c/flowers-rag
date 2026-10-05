"""Dobles de prueba compartidos: cliente de Google AI, vectores y helper de AST."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def imports_de(ruta) -> set[str]:
    """Módulos que un archivo importa de verdad (regla 03: sin backend local)."""
    arbol = ast.parse(Path(ruta).read_text(encoding="utf-8"))
    modulos: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos.update(alias.name for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module)
    return modulos


def vector(seed: int, dim: int = 8) -> list[float]:
    """Vector determinista y normalizado-ish para comparar similitudes."""
    return [round(1.0 / (seed + i + 1), 6) for i in range(dim)]


class FakeModels:
    def __init__(self, dim: int = 8, error: Exception | None = None, mismatch: bool = False):
        self.dim = dim
        self.error = error
        self.mismatch = mismatch
        self.calls: list[dict] = []
        self._seed = 0

    def embed_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        texts = contents if isinstance(contents, list) else [contents]
        n = len(texts) - 1 if self.mismatch else len(texts)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=vector(self._seed + i, self.dim)) for i in range(n)]
        )


class FakeClient:
    def __init__(self, dim: int = 8, error: Exception | None = None, mismatch: bool = False):
        self.models = FakeModels(dim=dim, error=error, mismatch=mismatch)

    @property
    def calls(self) -> list[dict]:
        return self.models.calls


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def fake_client_factory(monkeypatch):
    """Captura cómo se construye el cliente real de `google-genai`."""
    created: list[dict] = []

    def factory(**kwargs):
        created.append(kwargs)
        return FakeClient()

    monkeypatch.setattr("app.embed.genai.Client", factory)
    return created