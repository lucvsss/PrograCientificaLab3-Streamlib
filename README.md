<h1 align="center">📜 Biblical Text Mining — Dashboard interactivo</h1>
<h3 align="center">Laboratorio 3 · Programación Científica · Minor: Sistemas Inteligentes</h3>

<p align="center">
<img alt="UCN" src="https://img.shields.io/badge/Universidad_Católica_del_Norte-orange">
<img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
<img alt="Streamlit" src="https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
<img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
</p>

> Aplicación **cliente-servidor** para explorar el corpus bíblico (versión ASV, 31.103
> versículos). Una **API REST** en FastAPI concentra todo el procesamiento —filtrado,
> TF-IDF, PCA, Word2Vec, búsqueda y n-gramas— y una app **Streamlit** actúa solo como
> interfaz: recibe la interacción del usuario, llama a la API y muestra los resultados.
> La app nunca guarda la Biblia completa en memoria.

Este taller es la continuación del Laboratorio 2: reutiliza el núcleo de análisis
(`src/`) construido allí —TF-IDF y similitud del coseno propios, preprocesamiento,
modelos de n-gramas— y lo expone como servicio web.

---

## Arquitectura

```
┌────────────────────────┐        HTTP / JSON        ┌────────────────────────┐
│   API REST (FastAPI)    │  ◀────────────────────▶   │    App (Streamlit)     │
│                         │                           │                        │
│ • carga el corpus 1 vez │                           │ • dropdowns y controles│
│ • TF-IDF (src/ Lab 2)   │                           │ • gráficos Plotly      │
│ • PCA + Word2Vec        │                           │ • solo muestra lo que  │
│ • búsqueda coseno       │                           │   la API le devuelve   │
│ • n-gramas              │                           │                        │
└────────────────────────┘                           └────────────────────────┘
        :8000  /docs                                          :8501
```

## Estructura del proyecto

```
PrograCientificaLab3/
├── api/
│   ├── service.py          # motor de análisis (se construye 1 vez, reutiliza src/)
│   └── app.py              # FastAPI
├── streamlit_app/
│   ├── app.py              # interfaz de 4 secciones (dashboard, buscador, viz, generador)
│   └── client.py           # cliente HTTP hacia la API
├── src/                    # núcleo del Laboratorio 2 (TF-IDF, preprocesamiento, n-gramas, etc.)
├── data/                   # stopwords.json (+ CSV del corpus)
├── requirements.txt
└── README.md
```

## Instalación

```bash
git clone <URL-del-repositorio> PrograCientificaLab3
cd PrograCientificaLab3
pip install -r requirements.txt
```

## Cómo ejecutar

Se necesitan **dos terminales**: una para la API y otra para la app.

### Linux / macOS

**Terminal 1 — API** 
```bash
uvicorn api.app:app --reload
```
Queda en `http://localhost:8000` y la documentación interactiva en
`http://localhost:8000/docs`.

**Terminal 2 — App Streamlit**
```bash
streamlit run streamlit_app/app.py
```
Se abre en `http://localhost:8501`.

### Windows

Con **dos ventanas** de CMD o PowerShell abiertas en la carpeta del proyecto:

**Ventana 1 — API:**
```bat
uvicorn api.app:app --reload
```

**Ventana 2 — App:**
```bat
streamlit run streamlit_app/app.py
```

En caso de que `uvicorn` o `streamlit` no se reconozcan como comando, usa `python -m`:
```bat
python -m uvicorn api.app:app --reload
python -m streamlit run streamlit_app/app.py
```

## Funcionalidades

### 1. Dashboard (con filtros por testamento / libro / capítulo)
Versículos por libro, longitud promedio de versículos, palabras más frecuentes y nube
de palabras. **Todo el filtrado ocurre en la API**: la app solo pide el subconjunto que
está mostrando en el momento.

### 2. Buscador semántico
El usuario escribe una frase; la API la vectoriza con el mismo TF-IDF del corpus y
devuelve los versículos más parecidos por **similitud del coseno**. Ejemplos que
funcionan bien: `faith hope and love` → *1 Corinthians 13:13*; `the kingdom of heaven`
→ *Matthew 6:10*.

### 3. Visualizador PCA / Word2Vec (2D y 3D)
Proyección interactiva de los versículos. Se puede alternar entre:
- **TF-IDF + PCA**: parte de vectores dispersos de miles de dimensiones.
- **Word2Vec + PCA**: embeddings densos de 100 dimensiones proyectados con PCA.

La reducción de dimensionalidad se hace **en la API**; la app solo dibuja los puntos.

### 4. Generador de versículos
Selector de modelo (unigram · bigram · trigram · tetragram · pentagram), palabra inicial
y largo máximo. La API genera el texto con el modelo de n-gramas correspondiente. Incluye
un botón para **comparar** los cinco modelos con la misma semilla.

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET`  | `/meta` | Listas de testamentos, libros y capítulos (para poblar los filtros). |
| `GET`  | `/dashboard/versiculos-por-libro` | Conteo por libro. Filtros: `testamento`, `libro`, `capitulo`. |
| `GET`  | `/dashboard/longitud-promedio` | Largo medio de versículo por libro (mismos filtros). |
| `GET`  | `/dashboard/top-palabras?n=` | Palabras más frecuentes del subconjunto. |
| `GET`  | `/dashboard/wordcloud?n=` | Frecuencias para la nube de palabras. |
| `POST` | `/buscar` | `{query, k}` → versículos más similares. |
| `GET`  | `/proyeccion/pca?dim=2\|3` | PCA sobre TF-IDF. |
| `GET`  | `/proyeccion/word2vec?dim=2\|3` | PCA sobre embeddings Word2Vec. |
| `POST` | `/generar` | `{modelo, palabra_inicial, max_len}` → texto generado. |

Documentación completa y probador interactivo en `/docs`.

## Integrantes

<p align="center">
  <b>Lucas Munizaga</b> &nbsp;·&nbsp;
  <b>Sofía Bustos</b> &nbsp;·&nbsp;
  <b>Nicolás Rivas</b>
</p>

<p align="center"><sub>Universidad Católica del Norte · Escuela de Ingeniería, Coquimbo · 2026</sub></p>

