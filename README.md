# Sistema RAG para Consulta de Precios y Catálogo — Proyecto Final

Sistema de Generación Aumentada por Recuperación (RAG) construido con **Streamlit**, **FastAPI**, **ChromaDB** y **Google AI Studio** para responder consultas sobre catálogos y listas de precios sin alucinaciones.

## 🏗️ Arquitectura del Sistema
- **UI (Streamlit):** Cliente gráfico para cargar documentos y realizar preguntas con citas [n].
- **API (FastAPI):** Servidor HTTP que orquesta la ingesta, búsqueda vectorial y generación.
- **Base Vectorial (ChromaDB):** Almacenamiento persistente local en `./chroma`.
- **Embeddings & LLM (Google AI Studio):** Modelo `gemini-embedding-001` para vectores y `gemini-3.5-flash` para generación anclada.

## 🚀 Requisitos Previos e Instalación

### 1. Clonar el repositorio y crear entorno virtual
```bash
git clone <tu-repositorio>
cd proyecto_final/rag-app
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
