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

### 2. Configurar la clave de Google AI
```bash
cp .env.example .env      # y define GOOGLE_API_KEY (https://aistudio.google.com/apikey)
```

### 3. Arrancar la API (`:8000`) y la UI (`:8501`) en dos terminales
```bash
# Terminal 1 — FastAPI: chunking, embeddings de Google AI, ChromaDB y Gemini
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit: carga de documentos, chat y citas
streamlit run ui/streamlit_app.py --server.port 8501
```

La UI es un cliente HTTP de la API: no importa `app/`, ni `chromadb`, ni el SDK de
Google AI. Sube documentos en la barra lateral, pulsa **Indexar** y luego pregunta;
cada respuesta muestra debajo las citas `[n]` con su origen, su score y el texto del
chunk. Puedes apuntar la UI a otra API con `API_URL=http://otro-host:8000`.

## 🧪 Pruebas
```bash
python -m pytest -q
```
