# Sistema RAG — Catálogo de Flores y Listas de Precios

Sistema de **Generación Aumentada por Recuperación (RAG)** sobre un corpus real de
floristería: 19 documentos (6 Markdown y 13 PDF) con guías de cuidado, conservación,
listas de precios y términos de envío. Responde **solo** con la evidencia indexada y
se abstiene cuando no la hay.

Las cuatro piezas obligatorias están en el camino crítico: **Streamlit** (UI),
**FastAPI** (API), **ChromaDB** (índice vectorial persistente) y **Google AI Studio**
(embeddings con `gemini-embedding-001` y generación con `gemini-3.5-flash`).

---

## 📚 Índice

1. [Arquitectura](#-arquitectura)
2. [Instalación](#-instalación)
3. [Exportar la clave de Google AI](#-exportar-la-clave-de-google-ai)
4. [Ejecución (API + UI)](#-ejecución-api--ui)
5. [La regla de abstención](#-la-regla-de-abstención)
6. [Uso: indexar y preguntar](#-uso-indexar-y-preguntar)
7. [Variables de entorno](#-variables-de-entorno)
8. [Pruebas](#-pruebas)
9. [Estructura del proyecto](#-estructura-del-proyecto)
10. [Reporte de una página](#-reporte-de-una-página)

---

## 🏗️ Arquitectura

```
Usuario
  └── Streamlit  (:8501)          ui/streamlit_app.py — cliente HTTP, no importa app/
        └── HTTP JSON
              └── FastAPI  (:8000)   app/main.py — orquesta; único punto que toca Chroma y Google AI
                    ├── app/chunk.py    troceado en palabras con solape
                    ├── app/embed.py    embeddings con Google AI (el mismo modelo para todo)
                    ├── app/store.py    ChromaDB persistente + k-NN coseno
                    └── app/generate.py Gemini: respuesta anclada en la evidencia o abstención
```

Decisiones que sostienen el diseño:

| Decisión | Motivo |
|---|---|
| Un único motor de embeddings (`google-genai`) | La rúbrica prohíbe FastText, BERT local, bag-of-words y el embedder por defecto de ChromaDB. La colección se crea con `embedding_function=None` y los vectores se pasan siempre a Chroma (`app/store.py:62`). |
| El mismo modelo para documentos y preguntas | Si se mezclan modelos, la distancia coseno no significa nada (`EMBEDDING_MODEL`). |
| Recuperar **antes** de generar | `/query` siempre hace k-NN y solo después llama a Gemini con los chunks numerados `[1]`, `[2]`, … |
| Abstención = `200` de negocio | Una pregunta fuera de dominio no es un error: es un resultado de primera clase. |
| Chroma `PersistentClient` en `./chroma` | Reiniciar FastAPI no borra el índice. |
| La UI no importa `app/` | Verificado por una prueba que lee el AST del script: sus únicos imports son `httpx`, `streamlit`, `os`, `pathlib` y `typing`. |

---

## 🚀 Instalación

Requiere **Python 3.10 o superior** (probado con 3.13.9 en macOS).

```bash
git clone <tu-repositorio>
cd flowers-rag

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` fija las versiones con las que se probó el proyecto
(`streamlit`, `fastapi`, `chromadb`, `google-genai`, `pypdf`, `httpx`, `pytest`, …).

---

## 🔑 Exportar la clave de Google AI

La clave se obtiene gratis en
[Google AI Studio](https://aistudio.google.com/apikey) y **solo vive en el entorno
del servidor**: nunca se sube al repositorio (`.env` está en `.gitignore`).

```bash
cp .env.example .env
```

Edita `.env` y pega tu clave:

```dotenv
GOOGLE_API_KEY=AIzaSy...tu_clave...
```

Verifica que la API la ha leído:

```bash
curl -s localhost:8000/health | grep -i key
# "google_ai_key_configured": true
```

Si prefieres exportarla en la sesión en vez de usar `.env`:

```bash
export GOOGLE_API_KEY=AIzaSy...tu_clave...
```

> **Nota sobre la cuota.** Las llamadas a Google AI Studio tienen cuota por
> modelo y día. Si ves `503 google_ai_error` con `429 RESOURCE_EXHAUSTED`, no es un
> fallo del código: la clave agotó su cuota. Se puede comprobar en
> <https://ai.dev/rate-limit>.

---

## ▶️ Ejecución (API + UI)

Dos procesos, dos terminales. **La UI depende de que la API esté arriba.**

```bash
# Terminal 1 — FastAPI en :8000
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit en :8501
source .venv/bin/activate
streamlit run ui/streamlit_app.py --server.port 8501
```

| Servicio | URL |
|---|---|
| UI (Streamlit) | <http://localhost:8501> |
| API (FastAPI) | <http://localhost:8000> |
| Documentación OpenAPI | <http://localhost:8000/docs> |

CORS ya está configurado para `localhost:8501` (`ALLOWED_ORIGINS`), así que la UI
habla con la API sin tocar nada más. Para apuntar la UI a otra API:
`API_URL=http://otro-host:8000 streamlit run ui/streamlit_app.py`.

---

## 🚦 La regla de abstención

Esta es la parte central del proyecto: **el sistema prefiere no responder antes que
inventar**. Hay dos capas, y las dos devuelven `200` con `abstained: true`.

### Capa 1 — determinista: filtro por score (sin llamar al modelo)

`POST /query` recupera los `top_k` vecinos y **descarta los que no llegan al umbral
`min_score`** (por defecto `0.35`, `MIN_SCORE_DEFAULT`). El score es la similitud
coseno: `1 − distancia`, calculada por Chroma en `app/store.py:208`.

- Si **ningún** chunk alcanza el umbral → abstención inmediata, **Gemini no se
  invoca**, `citations: []` y `generation_ms: 0`:

```json
{
  "answer": "No tengo evidencia suficiente en el corpus indexado para responder a esa pregunta.",
  "abstained": true,
  "abstain_reason": "sin_evidencia",
  "citations": [],
  "generation_model": null
}
```

- Si **algún** chunk lo alcanza, secited esos chunks a Gemini numerados `[1]…[n]`.

### Capa 2 — el modelo: la evidencia no cubre la pregunta

Si hay chunks por encima del umbral pero ninguno responde, el prompt (`app/generate.py:34`)
obliga a Gemini a devolver **exactamente** esta frase y nada más:

> No encuentro en los documentos cargados información que responda a esa pregunta.

El backend la detecta y la marca como abstención, **conservando las citas** para que se
vea por qué se abstuvo:

```json
{
  "abstained": true,
  "abstain_reason": "modelo_abstuvo",
  "citations": [ { "index": 1, "score": 0.47, "...": "..." } ]
}
```

### El prompt que hace posible la regla

`app/generate.py` le da a Gemini cuatro instrucciones fijas:

1. Usa solo los fragmentos `[1]..[n]`, sin conocimiento propio.
2. Cita cada afirmación con `[n]` usando esos mismos números.
3. Responde siempre en español.
4. Si la evidencia no responde, devuelve la frase de abstención exacta.

Y la API garantiza el contrato de citas: cada `[n]` del texto corresponde a
`citations[n - 1]`, con su `source`, su `text` y su `score`.

### Qué NO es una abstención

Una abstención es un `200`. Los errores de infraestructura van por otro camino, con
códigos estables para que la UI decida el mensaje:

| Situación | HTTP | `code` |
|---|---|---|
| Índice vacío (nunca se ingirió nada) | `503` | `empty_index` |
| Falta `GOOGLE_API_KEY` | `503` | `missing_google_api_key` |
| Google AI no responde a tiempo | `504` | `google_ai_timeout` |
| Cuota o error de Google AI | `503` | `google_ai_error` |
| ChromaDB inaccesible | `503` | `chroma_unavailable` |
| Pregunta en blanco | `400` | `empty_question` |
| Fallo inesperado | `500` | `internal_error` |

---

## 💬 Uso: indexar y preguntar

### Desde la UI

1. **Barra lateral** → *Documentos* → sube uno o varios `.md`, `.txt`, `.pdf` o `.csv`.
   Puedes ajustar *Palabras por chunk* (320), *Solape* (64) y *Vaciar el índice*.
2. Pulsa **Indexar**. Se escribe el archivo en `data/uploads/` y la UI llama a
   `POST /ingest`. Verás un resumen: *"Indexados N chunks de M documentos"*, y los
   omitidos con su motivo.
3. **Chat principal**: escribe la pregunta. Cada respuesta muestra el texto de Gemini
   y, debajo, el panel **Fuentes (n)** con `[n]`, documento de origen, score y el
   texto completo del chunk. Las respuestas quedan en el historial de la sesión.

### Desde `curl` (mismo contrato que `/docs`)

Indexar el corpus de ejemplo:

```bash
curl -s -X POST localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"paths": ["data/"], "chunk_size": 320, "chunk_overlap": 64}'
```

Respuesta real (respuesta completa; `chunks_skipped` cuenta chunks sueltos y es `0`
porque los descartes son a nivel de documento):

```json
{
  "documents_processed": 17,
  "chunks_indexed": 44,
  "chunks_skipped": 0,
  "collection": "flowers_catalog",
  "embedding_model": "gemini-embedding-001",
  "sources": ["data/01_cuidados_y_temporadas_rosas.md", "…"],
  "skipped": [
    { "source": "data/precios_proveedores/LISTA DE PRECIOS COMPRA 16….pdf",
      "reason": "google_ai_error" }
  ],
  "duration_ms": 8339
}
```

Los 3 documentos omitidos son los que la **cuota de Google AI** rechazó durante esa
tanda: `documents_processed` cuenta los que sí se indexaron, y `skipped` dice cuáles
quedaron fuera y por qué. Reejecuta `POST /ingest` cuando la cuota se restablezca.

Pregunta en dominio:

```bash
curl -s -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "¿Cómo hay que cortar el tallo de una rosa?", "top_k": 3}'
```

```json
{
  "answer": "Corta el tallo en diagonal a unos 45° para que no se obstruya la savia [1]…",
  "abstained": false,
  "abstain_reason": null,
  "citations": [
    { "index": 1, "id": "data/01_….pdf#chunk-004", "score": 0.83,
      "source": "data/01_cuidados_y_temporadas_rosas.md", "text": "…" }
  ],
  "generation_model": "gemini-3.5-flash",
  "retrieval_ms": 669, "generation_ms": 23342
}
```

**Pregunta imposible** (no aparece en ningún documento del corpus):

```bash
curl -s -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "¿Quién ganó el Mundial de fútbol de 1998 y con qué marcador?"}'
```

El sistema **no inventa**: se abstiene con `abstained: true`. Si los chunks abiertos
superan `min_score` pero no cubren la pregunta, devuelve `abstain_reason:
"modelo_abstuvo"` conservando las citas; si ni siquiera superan el umbral,
`"sin_evidencia"` sin llamar a Gemini.

---

## ⚙️ Variables de entorno

Todas en `.env` (copia de `.env.example`) y con valor por defecto en el código
(`app/settings.py`).

| Variable | Default | Para qué |
|---|---|---|
| `GOOGLE_API_KEY` | *(vacía)* | **Obligatoria.** embeddings y generación. |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Vectoriza chunks **y** preguntas. |
| `GENERATION_MODEL` | `gemini-3.5-flash` | Redacta la respuesta. |
| `CHROMA_PATH` | `./chroma` | Persistencia del índice. |
| `CHROMA_COLLECTION` | `flowers_catalog` | Nombre de la colección. |
| `CHUNK_SIZE` | `320` | Palabras por chunk (rango rubricado: 200–400). |
| `CHUNK_OVERLAP` | `64` | Palabras de solape (rango: 40–80). |
| `TOP_K_DEFAULT` | `3` | Vecinos recuperados por defecto. |
| `MIN_SCORE_DEFAULT` | `0.35` | Umbral de evidencia: por debajo, abstención. |
| `MAX_FILE_BYTES` | `26214400` | Tope por archivo (25 MB). |
| `EMBED_BATCH_SIZE` | `16` | Textos por llamada a Google AI (cuota). |
| `GOOGLE_TIMEOUT_SECONDS` | `30` | Timeout de las llamadas a Google AI. |
| `ALLOWED_ORIGINS` | `localhost:8501, 127.0.0.1:8501` | CORS de la UI. |
| `API_URL` | `http://localhost:8000` | Solo UI: dónde vive la API. |
| `UPLOADS_DIR` | `./data/uploads` | Solo UI: dónde se guardan los archivos subidos (se crea sola). |

> **Cambiar de modelo de embeddings exige reindexar.**
> `gemini-embedding-001` devuelve 3072 dimensiones. Si mezclas modelos, `/query`
> responde `503 dimension_mismatch` (comprobado en `app/store.py:235`). Solución:
> borrar `chroma/` y volver a indexar.

---

## 🧪 Pruebas

```bash
source .venv/bin/activate
python -m pytest -q          # 94 pruebas
```

Ninguna prueba sale a la red: Google AI está doblado con `SimpleNamespace` y ChromaDB
usa un directorio temporal. La UI se prueba con `streamlit.testing.v1.AppTest`, que
ejecuta el script real en proceso contra un servidor HTTP de mentira en un hilo.

Qué cubren, por regla de la rúbrica:

| Fichero | Qué verifica |
|---|---|
| `tests/test_embed.py` | Vectores de Google AI, lotes, dimensión, errores y timeouts. |
| `tests/test_store.py` | Persistencia, k-NN coseno, metadatos, `dimension_mismatch`, borrado por `source`. |
| `tests/test_api.py` | `/ingest`: chunking, rutas, omitidos con motivo, `413`. |
| `tests/test_query.py` | `/query`: mapa `[n]` ↔ `citations[n-1]`, abstención por score y por modelo, `503`/`504`, filtro por `source`. |
| `tests/test_health.py` | `/health`, CORS y el `500` uniforme (`internal_error`). |
| `tests/test_ui.py` | Aislamiento por AST, subida real de archivos, chat, citas y estados de error. |

---

## 📁 Estructura del proyecto

```
flowers-rag/
├── README.md
├── requirements.txt              # versiones probadas
├── .env.example                  # plantilla (GOOGLE_API_KEY vacía)
├── .env                          # tu clave — NO se sube (está en .gitignore)
├── data/                         # corpus: 6 .md + 13 PDF de floristería
│   └── uploads/                  # lo que sube la UI (ignorado por git)
├── chroma/                       # índice persistente (ignorado por git)
├── docs/
│   ├── openapi_spec.yaml         # contrato HTTP: precede a código y pruebas
│   └── rubrica.md                # enunciado del proyecto
├── app/
│   ├── main.py                   # FastAPI: /health, /ingest, /query
│   ├── models.py                 # esquemas (espejo de components.schemas)
│   ├── settings.py               # configuración por entorno
│   ├── chunk.py                  # extracción (.md/.txt/.csv/.pdf) y troceado
│   ├── embed.py                  # embeddings con Google AI
│   ├── store.py                  # ChromaDB: alta, k-NN, borrado por fuente
│   └── generate.py               # Gemini: respuesta anclada + abstención
├── ui/
│   └── streamlit_app.py          # carga, chat, citas, scores
└── tests/                        # 94 pruebas
```

---

## 📝 Reporte de una página

**Dominio y corpus.** Floristería: cuidado y conservación de flores, precios y
términos de envío. **19 documentos** (6 Markdown propios + 13 PDF reales de
floristería) con **13 612 palabras**, que con la configuración por defecto dan
**60 chunks** en la colección `flowers_catalog`. Las listas de precios de proveedor
(`data/precios_proveedores/`) son las piezas más largas, con 8 chunks cada una. Un
documento que no se puede extraer se cuenta en `skipped` con su motivo en vez de
fingir que se indexó.

**Particionado.** 320 palabras por chunk con 64 de solape (20 %). Se cuenta en
**palabras**, no caracteres, porque los tokens cambian con el idioma y la versión del
modelo. El solape existe para que un dato que cae justo en la frontera entre dos trozos
siga completo en alguno de los dos: con `stride = chunk_size − chunk_overlap` no se
pierde ni una palabra al concatenar.

**Cómo se decide abstenerse.** En dos capas. La determinista es un umbral de
similitud coseno: los vecinos k-NN que no llegan a `min_score = 0.35` se descartan, y
si no queda ninguno la API se abstiene **sin llamar a Gemini** (`sin_evidencia`). Si
los hay, se los pasa a Gemini con la instrucción de responder solo con esa evidencia y,
si no cubre, repetir la frase de abstención exacta; el backend la detecta y la marca
`modelo_abstuvo` conservando las citas. En ambos casos es un `200` con
`abstained: true`, no un error.

**Qué sale de Google AI y qué hace Chroma.** Google AI Studio hace las dos cosas de
modelo, con modelos distintos: `gemini-embedding-001` vectoriza (3072 dimensiones,
lotes de 16 para respetar la cuota) y `gemini-3.5-flash` redacta. ChromaDB solo
persiste y calcula distancias: recibe los vectores ya calculados
(`embeddings=` en `add` y `query`) con `embedding_function=None`, en coseno, en
`./chroma`. La API es la única capa que habla con Google AI y con Chroma; la UI solo
habla con la API.

**Estado y limitaciones conocidas.** La cuota diaria de `gemini-3.5-flash` es el
cuello de botella real: con la cuota agotada la ingesta funciona pero `/query`
devuelve `503 google_ai_error` con `429 RESOURCE_EXHAUSTED`. El índice actual se
construyó con el modelo vigente; cambiar de modelo de embeddings exige borrar
`chroma/` y reindexar. `POST /ingest` acepta rutas del servidor (modo JSON), que es lo
que usa la UI: los archivos subidos se escriben en `data/uploads/` y se envían como
rutas, de modo que la UI no necesita `multipart/form-data`. Quedan declarados en
`docs/openapi_spec.yaml` pero **sin implementar** los dos endpoints opcionales del
reto (`GET /index` y `DELETE /index/sources/{source}`): la lógica de borrado por
documento ya existe y está probada en `app/store.py:113`, pero no se expone por
HTTP.