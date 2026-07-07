import os
import requests
import streamlit as st

API_URL = os.environ.get("BTM_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 60


def _get(path, params=None):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(path, payload):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner=False)
def get_meta():
    return _get("/meta")


def ping():
    try:
        _get("/")
        return True
    except Exception:
        return False

#DASHBOARD
def versiculos_por_libro(**f):
    return _get("/dashboard/versiculos-por-libro", _clean(f))


def longitud_promedio(**f):
    return _get("/dashboard/longitud-promedio", _clean(f))


def top_palabras(n=20, **f):
    return _get("/dashboard/top-palabras", _clean({**f, "n": n}))


def wordcloud(n=100, **f):
    return _get("/dashboard/wordcloud", _clean({**f, "n": n}))


#BUSCADOR
def buscar(query, k=10):
    return _post("/buscar", {"query": query, "k": k})


#PROYECCIONES
def proyeccion(tecnica="pca", dim=2, cap_por_libro=40):
    return _get(f"/proyeccion/{tecnica}", {"dim": dim, "cap_por_libro": cap_por_libro})


#GENERADOR
def generar(modelo="bigram", palabra_inicial=None, max_len=20):
    return _post("/generar", {
        "modelo": modelo, "palabra_inicial": palabra_inicial, "max_len": max_len,
    })


def _clean(d):
    return {k: v for k, v in d.items() if v is not None and v != ""}
