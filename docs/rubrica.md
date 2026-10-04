# Proyecto final — Sistema RAG (Streamlit + FastAPI + ChromaDB + Google AI)

## Contexto

**RAG** (generación aumentada por recuperación) responde con evidencia
recuperada, no solo con lo que el modelo “recuerda”. El ciclo es:

```
incrustar → indexar → recuperar top-k → generar respuesta anclada
```

Este proyecto final consiste en **desarrollar un sistema RAG**. En este
proyecto **sí vas a programar**. Trabajas en una carpeta nueva, por ejemplo
`proyecto_final/rag-app/` (puedes nombrarla distinto si el README lo
documenta).

Las notebooks `RAG/Notebooks/` (FastText + BERT extractivo) muestran una
variante del mismo ciclo. Aquí la recuperación va contra **ChromaDB** y los
vectores los genera **Google AI**, no FastText.

## Objetivo

Implementar un sistema **RAG** completo, con:

| Capa | Tecnología | Rol |
|---|---|---|
| UI | **Streamlit** | Cargar documentos, preguntar y ver la respuesta con citas |
| API | **FastAPI** | Ingestar, consultar e informar el estado del índice |
| Índice | **ChromaDB** | Guardar chunks + embeddings y devolver los más similares |
| Embeddings | **Google AI** | Convertir cada chunk y cada pregunta en un vector |

La interfaz **no** habla con Chroma ni con Google AI en silencio: todo pasa
por la API. Streamlit es un cliente HTTP de FastAPI.

## Arquitectura pedida

```
Usuario
  └── Streamlit (puerto 8501)
        └── HTTP JSON
              └── FastAPI (puerto 8000)
                    ├── Google AI  → embeddings (obligatorio)
                    ├── ChromaDB   → persistencia y k-NN
                    └── Google AI  → generación de la respuesta (Gemini)
```

1. El usuario sube documentos (o apunta a una carpeta) en Streamlit.
2. FastAPI parte el texto en chunks, pide el embedding a Google AI y lo
   guarda en ChromaDB.
3. El usuario escribe una pregunta. FastAPI incrusta la pregunta con el
   **mismo** modelo de embeddings, recupera los `top-k` chunks y le pide a
   Gemini una respuesta **solo** con esa evidencia.
4. Streamlit muestra la respuesta, las citas `[n]` y los chunks usados
   (texto, origen, score).

## Stack obligatorio

No sustituyas estas cuatro piezas. Puedes añadir librerías de apoyo
(`httpx`, `pypdf`, `python-dotenv`, `uvicorn`, etc.).

1. **Streamlit** — única UI. No entregues solo un CLI ni un HTML suelto.
2. **FastAPI** — única API. Endpoints documentados (OpenAPI / `/docs`).
3. **ChromaDB** — única base vectorial. Persistente en disco (no un dict en
   RAM que se pierde al reiniciar).
4. **Google AI** — fuente posible de **embeddings**. Usa la API de Google AI
   Studio / Gemini (`google-genai` o el SDK equivalente). Modelo de
   embeddings, por ejemplo `text-embedding-004` o `gemini-embedding-001`.

La clave va en un archivo `.env` (`GOOGLE_API_KEY=...`) que **no** se
sube al repositorio. En el README indica cómo obtenerla en
[Google AI Studio](https://aistudio.google.com/apikey).

Para **generar** la respuesta usa **Gemini** (mismo proveedor). El modelo
debe escribir en español, citar `[n]` y **abstenerse** si no hay evidencia.
No armes la respuesta concatenando chunks a mano.

## Corpus

Tú eliges el dominio, con estas reglas:

- Al menos **5 documentos** distintos (PDF, Markdown o texto). No basta un
  solo párrafo repetido.
- Tema coherente (apuntes del curso, un reglamento, un conjunto de papers,
  un manual, Wikipedia de un tema, etc.).
- Suficiente texto para que el chunking importe: del orden de **varios
  miles de palabras** en total, no cinco tuits.
- Incluye **al menos una pregunta imposible** de responder con el corpus
  (algo que no aparezca en ningún documento): el sistema debe abstenerse,
  no inventar.

No uses material con derechos que no puedas compartir en la entrega.

## Archivos a crear

Sugerencia de estructura (puedes variar nombres si el README lo explica):

```
proyecto_final/rag-app/
  README.md
  requirements.txt
  .env.example
  data/                 # corpus de ejemplo (sin secretos)
  chroma/               # persistencia local (en .gitignore)
  app/
    main.py             # FastAPI: /health, /ingest, /query
    chunk.py            # partición en chunks con overlap
    embed.py            # cliente de embeddings de Google AI
    store.py            # ChromaDB: alta, consulta top-k
    generate.py         # Gemini: respuesta anclada + abstenerse
  ui/
    streamlit_app.py    # carga, chat, citas, scores
```

Los vectores del índice salen de **Google AI**. No uses FastText, BERT
local ni bolsa de palabras como backend de producción.

## Requisitos

### API (FastAPI)

1. `GET /health` — confirma que la API vive y, si puedes, que Chroma está
   accesible.
2. `POST /ingest` — recibe archivos o rutas, chunkifica, incrusta con
   Google AI y persiste en Chroma. Responde cuántos documentos y cuántos
   chunks indexó.
3. `POST /query` — cuerpo JSON con `question` y, opcionalmente, `top_k`.
   Responde al menos:

   - `answer` (texto generado),
   - `citations` (lista de chunks: `id` o índice, `source`, `text`, `score`),
   - `abstained` (booleano: no había evidencia suficiente).

4. La API no debe devolver 500 por una pregunta fuera de dominio: debe
   abstenerse con un mensaje claro.
5. CORS o misma máquina: Streamlit en `localhost:8501` debe poder llamar a
   `localhost:8000`.

### Índice (ChromaDB)

6. Colección persistente (ruta local, p. ej. `chroma/`). Reiniciar FastAPI
   **no** borra el índice.
7. Cada chunk guarda metadatos: `source` (archivo), opcionalmente `title` y
   posición (página o índice de chunk).
8. La consulta usa el embedding de la pregunta (Google AI) y `top-k`
   vecinos. `k` por defecto razonable (2–5), configurable.

### Embeddings (Google AI)

9. El **mismo** modelo incrusta documentos y preguntas. Si mezclas
   modelos, el k-NN no significa nada.
10. No reimplementes embeddings con FastText, BERT local ni bag-of-words
    para el índice de producción.

### Generación

11. El prompt incluye los chunks numerados `[1]`, `[2]`, … y la instrucción
    de responder **solo** con esa evidencia, en español, con citas.
12. Umbral o criterio de abstención (score mínimo, o el propio modelo si
    el contexto no cubre la pregunta). Documenta la regla en el README.
13. Si te abstienes, `answer` lo dice explícitamente (p. ej. “no tengo
    evidencia suficiente”) y no rellenas con conocimiento paramétrico.

### UI (Streamlit)

14. Página para **cargar** documentos e indexarlos (llama a `/ingest`).
15. Caja de pregunta (o chat). Al enviar, llama a `/query`.
16. Muestra la respuesta, las citas y los chunks (origen + score). No basta
    un `print` en la terminal de FastAPI.
17. Estados vacíos y de error visibles: API caída, clave ausente, corpus
    vacío, pregunta en blanco.

### Ingeniería

18. `requirements.txt` con versiones que te funcionaron.
19. `.env.example` con `GOOGLE_API_KEY=` vacío. `.env` y `chroma/` en
    `.gitignore`.
20. README con cómo crear el venv, exportar la clave, levantar **API y UI**
    y probar una pregunta.

## Pasos sugeridos

1. Crea el venv y un FastAPI mínimo (`GET /health`) con Uvicorn.

2. Implementa `chunk.py` (tamaño y overlap configurables, p. ej. 200–400
   palabras y 40–80 de solape). Empieza con `.txt` / `.md`; el PDF puede
   ir después.

3. En `embed.py`, un único cliente que manda texto a Google AI y devuelve
   `list[float]`. Prueba con dos frases parecidas y una ajena: las
   primeras deben quedar más cercanas (coseno).

4. Conecta Chroma: `add` de chunks+vectores+metadatos y `query` por
   vector. Verifica persistencia reiniciando el proceso.

5. `POST /ingest` y `POST /query` sin generación: de momento la “respuesta”
   puede ser el chunk de mayor score. Cuando eso sea estable, enchufa
   Gemini en `generate.py`.

6. Streamlit: primero la pregunta contra una API ya indexada a mano;
   después el formulario de carga.

7. Prueba tres preguntas del dominio y **una** fuera de dominio. Las tres
   primeras deben citar fuentes reales; la cuarta debe abstenerse.

## Criterios de aceptación

- Streamlit, FastAPI, ChromaDB y embeddings de Google AI están **los
  cuatro** en el camino crítico (no son imports de adorno).
- Streamlit no accede a Chroma ni a Google AI salvo a través de FastAPI.
- Ingerir ≥ 5 documentos y preguntar produce una respuesta en español con
  citas `[n]` que apuntan a chunks visibles en la UI.
- Reiniciar la API conserva el índice (Chroma persistente).
- Una pregunta sin evidencia se abstiene; no alucina un dato que no está
  en el corpus.
- `GET /docs` de FastAPI muestra `/health`, `/ingest` y `/query`.
- No hay claves en el git. El README basta para que otra persona levante
  el sistema con su propia `GOOGLE_API_KEY`.

## Entrega

1. Código en `proyecto_final/`, con
   README, `requirements.txt` y `.env.example`.
2. El corpus de ejemplo (o un enlace + instrucciones de descarga).
3. Evidencias (capturas o un short screencast):
   - Streamlit mostrando una respuesta **con citas** y scores,
   - la misma pregunta en `/docs` o con `curl`/`httpx` contra FastAPI,
   - una pregunta **fuera de dominio** en la que el sistema se abstiene.
4. Un reporte corto (una página) que responda:
   - Dominio y tamaño del corpus (documentos, chunks, modelo de embedding).
   - Cómo particionaste (tamaño, overlap) y por qué.
   - Cómo decides abstenerte.
   - Qué sale de Google AI (embeddings vs. generación) y qué hace Chroma.

## Reto opcional

- Filtro por `source` (consultar solo los chunks de un archivo).
- Borrar o reindexar un documento sin reconstruir toda la colección.
- Histórico de preguntas en la sesión de Streamlit.
- Docker Compose: un servicio para la API y otro para la UI.

## Pistas

- El orden importa: **primero** recuperas, **después** generas. Si le das
  a Gemini la pregunta sin chunks, dejas de hacer RAG.
- Embeddings y generación son modelos distintos en la misma API. No uses
  un modelo de chat para vectorizar.
- Chroma puede incrustar con su propio embedder por defecto: **no lo uses**
  para este proyecto. Tú le pasas los vectores que ya calculó Google AI
  (`embeddings=` en `add` / `query`).
- Si `/ingest` se siente lento, el cuello suele ser la cuota de la API:
  manda textos en lotes pequeños y respeta el tamaño máximo del modelo.
- PDFs escaneados (solo imagen) no tienen texto: o extraes con OCR o no
  los cuentes como documento.
- `uvicorn app.main:app --reload --port 8000` y, en otra terminal,
  `streamlit run ui/streamlit_app.py`.
- Un umbral de score mínimo (`min_score`) es una forma simple de
  abstenerse cuando ningún chunk se parece a la pregunta.