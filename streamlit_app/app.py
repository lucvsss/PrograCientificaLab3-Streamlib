import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

sys.path.append(str(Path(__file__).resolve().parent))
import client

st.set_page_config(page_title="Biblical Text Mining", layout="wide")

if not client.ping():
    st.error(
        "No se pudo conectar con la API\n\n"
        "Intenta con:  uvicorn api.app:app --reload"
    )
    st.stop()

meta = client.get_meta()

# ---------------- menu lateral ----------------
st.sidebar.title("Biblical Text Mining")
st.sidebar.caption("Programacion Cientifica - Laboratorio 3")
seccion = st.sidebar.radio(
    "Seccion",
    ["Inicio", "Dashboard", "Buscador semantico",
     "Visualizador PCA / Word2Vec", "Generador de versiculos"],
)
st.sidebar.divider()
st.sidebar.caption("Corpus: " + str(meta["n_versiculos"]) + " versiculos / " + str(meta["n_libros"]) + " libros")
st.sidebar.caption("Vocabulario TF-IDF: " + str(meta["vocabulario"]) + " terminos")


def filtros_sidebar(clave):
    st.sidebar.subheader("Filtros")

    test = st.sidebar.selectbox("Testamento", ["(todos)"] + meta["testamentos"], key="t_" + clave)
    test_val = None if test == "(todos)" else test

    libros = [l["libro"] for l in meta["libros"] if test_val is None or l["testamento"] == test_val]
    libro = st.sidebar.selectbox("Libro", ["(todos)"] + libros, key="l_" + clave)
    libro_val = None if libro == "(todos)" else libro

    cap_val = None
    if libro_val is not None:
        n_caps = meta["capitulos_por_libro"].get(libro_val, 1)
        cap = st.sidebar.selectbox("Capitulo", ["(todos)"] + list(range(1, n_caps + 1)), key="c_" + clave)
        cap_val = None if cap == "(todos)" else int(cap)

    return {"testamento": test_val, "libro": libro_val, "capitulo": cap_val}


#INICIO
if seccion == "Inicio":
    st.title("Biblical Text Mining")
    st.write("Exploracion interactiva del corpus biblico (ASV) con arquitectura cliente-servidor.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Versiculos", meta["n_versiculos"])
    col2.metric("Libros", meta["n_libros"])
    col3.metric("Terminos (vocab.)", meta["vocabulario"])

    st.divider()
    st.write(
        "Esta aplicacion consume una API REST (FastAPI) que concentra todo el "
        "procesamiento del corpus. La interfaz solo pide y muestra; nunca guarda "
        "la Biblia completa en memoria."
    )
    st.markdown("""
    - **Dashboard**: versiculos por libro, longitud promedio, palabras frecuentes y nube de palabras (con filtros).
    - **Buscador semantico**: versiculos mas parecidos a una frase (similitud del coseno sobre TF-IDF).
    - **Visualizador PCA / Word2Vec**: proyeccion 2D/3D de los versiculos.
    - **Generador de versiculos**: texto generado con modelos de n-gramas.
    """)

#DASHBOARD
elif seccion == "Dashboard":
    st.title("Dashboard")
    f = filtros_sidebar("dash")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Versiculos por libro")
        data = client.versiculos_por_libro(**f)
        if data:
            df = pd.DataFrame(data).head(25).set_index("libro")
            st.bar_chart(df["n_versiculos"], horizontal=True)
        else:
            st.info("Sin datos para este filtro.")

    with col2:
        st.subheader("Longitud promedio de versiculos")
        data = client.longitud_promedio(**f)
        if data:
            df = pd.DataFrame(data).head(25).set_index("libro")
            st.bar_chart(df["longitud_promedio"], horizontal=True)
        else:
            st.info("Sin datos para este filtro.")

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Palabras mas frecuentes")
        top = client.top_palabras(n=15, **f)
        if top:
            df = pd.DataFrame(top).set_index("palabra")
            st.bar_chart(df["frecuencia"], horizontal=True)

    with col4:
        st.subheader("Nube de palabras")
        wc_data = client.wordcloud(n=120, **f)
        if wc_data:
            freqs = {d["palabra"]: d["frecuencia"] for d in wc_data}
            wc = WordCloud(width=700, height=400, background_color="white").generate_from_frequencies(freqs)
            st.image(wc.to_array(), use_container_width=True)

#BUSCADOR
elif seccion == "Buscador semantico":
    st.title("Buscador semantico")
    st.write("Escribe una frase y la API devuelve los versiculos mas parecidos (similitud del coseno sobre TF-IDF).")

    query = st.text_input("Frase a buscar", value="love peace and faith")
    k = st.slider("Cantidad de resultados", 3, 25, 10)

    if query.strip():
        resp = client.buscar(query, k=k)
        res = resp["resultados"]
        if res:
            filas = []
            for r in res:
                filas.append({
                    "Referencia": r["libro"] + " " + str(r["capitulo"]) + ":" + str(r["versiculo"]),
                    "Versiculo": r["texto"],
                    "Similitud": r["similitud"],
                })
            st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
        else:
            st.warning("Ninguna palabra de la consulta esta en el vocabulario. Prueba con otros terminos (en ingles).")

#VISUALIZADOR
elif seccion == "Visualizador PCA / Word2Vec":
    st.title("Visualizador PCA / Word2Vec")
    st.write("Proyeccion de los versiculos en 2D/3D. La reduccion de dimensionalidad se calcula en la API.")

    col1, col2, col3 = st.columns(3)
    tecnica = col1.radio("Representacion", ["TF-IDF + PCA", "Word2Vec + PCA"])
    dim = col2.radio("Dimensiones", [2, 3], format_func=lambda d: str(d) + "D")
    color_por = col3.selectbox("Colorear por", ["testamento", "genero"])
    cap = st.slider("Versiculos por libro (muestra)", 10, 120, 40, step=10)

    tec_key = "pca" if tecnica.startswith("TF-IDF") else "word2vec"
    data = client.proyeccion(tecnica=tec_key, dim=dim, cap_por_libro=cap)
    df = pd.DataFrame(data["puntos"])

    var = data["varianza_explicada"]
    st.caption("Varianza explicada: " + " / ".join("PC" + str(i + 1) + " " + str(v) + "%" for i, v in enumerate(var)))

    if dim == 3:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color=color_por, hover_data=["ref", "texto"])
    else:
        fig = px.scatter(df, x="x", y="y", color=color_por, hover_data=["ref", "texto"])
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "TF-IDF + PCA parte de vectores dispersos de miles de dimensiones, por eso explica poca "
        "varianza. Word2Vec + PCA parte de vectores densos de 100 dimensiones y explica mucha mas."
    )

#GENERADOR
elif seccion == "Generador de versiculos":
    st.title("Generador de versiculos")
    st.write("Genera texto con modelos de n-gramas entrenados sobre el corpus. La generacion ocurre en la API.")

    col1, col2, col3 = st.columns(3)
    modelo = col1.selectbox("Modelo", ["unigram", "bigram", "trigram", "tetragram", "pentagram"], index=1)
    palabra = col2.text_input("Palabra inicial (opcional)", value="god")
    max_len = col3.slider("Largo maximo", 5, 40, 20)

    if st.button("Generar"):
        resp = client.generar(modelo=modelo, palabra_inicial=palabra.strip() or None, max_len=max_len)
        st.success(resp["texto"])
        st.caption("Modelo: " + resp["modelo"] + " (n=" + str(resp["n"]) + ")")

    st.divider()
    st.subheader("Comparar los modelos")
    st.write("Genera con la misma palabra inicial para ver como cambia la coherencia segun n.")

    if st.button("Comparar unigram a pentagram"):
        filas = []
        for m in ["unigram", "bigram", "trigram", "tetragram", "pentagram"]:
            r = client.generar(modelo=m, palabra_inicial=palabra.strip() or None, max_len=max_len)
            filas.append({"Modelo": m, "n": r["n"], "Texto generado": r["texto"]})
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
        st.caption("A mayor n, el texto suele ser mas coherente pero mas corto y repetitivo.")