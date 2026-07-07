from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.service import BibleAnalysisService

servicio = BibleAnalysisService(data_dir="data")

app = FastAPI(
    title="Biblical Text Mining API",
    description="API REST que procesa y analiza el corpus bíblico (Lab. 3).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class BuscarRequest(BaseModel):
    query: str
    k: int = 10


class GenerarRequest(BaseModel):
    modelo: str = "bigram"
    palabra_inicial: Optional[str] = None
    max_len: int = 20


#RAIZ
@app.get("/")
def raiz():
    """Endpoint de salud: la app Streamlit lo usa para saber si la API está viva."""
    return {"status": "ok", "servicio": "Biblical Text Mining API"}


@app.get("/meta")
def meta():
    """Metadatos para poblar los filtros (testamentos, libros, capítulos, totales)."""
    return servicio.meta()


#DASHBOARD
@app.get("/dashboard/versiculos-por-libro")
def versiculos_por_libro(
    testamento: Optional[str] = None,
    libro: Optional[str] = None,
    capitulo: Optional[int] = None,
):
    return servicio.versiculos_por_libro(
        testamento=testamento, libro=libro, capitulo=capitulo
    )


@app.get("/dashboard/longitud-promedio")
def longitud_promedio(
    testamento: Optional[str] = None,
    libro: Optional[str] = None,
    capitulo: Optional[int] = None,
):
    return servicio.longitud_promedio(
        testamento=testamento, libro=libro, capitulo=capitulo
    )


@app.get("/dashboard/top-palabras")
def top_palabras(
    n: int = Query(20, ge=1, le=200),
    testamento: Optional[str] = None,
    libro: Optional[str] = None,
    capitulo: Optional[int] = None,
):
    return servicio.top_palabras(
        n=n, testamento=testamento, libro=libro, capitulo=capitulo
    )


@app.get("/dashboard/wordcloud")
def wordcloud(
    n: int = Query(100, ge=1, le=400),
    testamento: Optional[str] = None,
    libro: Optional[str] = None,
    capitulo: Optional[int] = None,
):
    return servicio.top_palabras(
        n=n, testamento=testamento, libro=libro, capitulo=capitulo
    )


#BUSCADOR
@app.post("/buscar")
def buscar(req: BuscarRequest):
    resultados = servicio.buscar(req.query, k=req.k)
    return {"query": req.query, "k": req.k, "resultados": resultados}


#PROYECCIONES
@app.get("/proyeccion/{tecnica}")
def proyeccion(
    tecnica: str,
    dim: int = Query(2, ge=2, le=3),
    cap_por_libro: int = Query(40, ge=1, le=500),
):
    if tecnica == "pca":
        return servicio.proyeccion_pca(dim=dim, cap_por_libro=cap_por_libro)
    if tecnica == "word2vec":
        return servicio.proyeccion_word2vec(dim=dim, cap_por_libro=cap_por_libro)
    return {"error": f"Técnica desconocida: '{tecnica}'. Usa 'pca' o 'word2vec'."}


#GENERAR TEXTO
@app.post("/generar")
def generar(req: GenerarRequest):
    return servicio.generar(
        modelo=req.modelo,
        palabra_inicial=req.palabra_inicial,
        max_len=req.max_len,
    )
