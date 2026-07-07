#MOTOR ANALISIS CORPUS BIBLICO
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

from src.data_loader import cargar_dataset
from src.preprocessing import TextPreprocessor
from src.tfidf import TFIDFVectorizer
from src.ngram_model import NGramModel

MAX_TERMINOS = int(os.environ.get("BTM_MAX_FEATURES", "6000"))
DIM_W2V = int(os.environ.get("BTM_W2V_DIM", "100"))


class BibleAnalysisService:

    def __init__(self, data_dir="data"):
        carpeta = Path(data_dir)

        print("[servicio] Cargando corpus...")
        df_original = cargar_dataset(
            carpeta / "t_asv.csv",
            carpeta / "key_english.csv",
            carpeta / "key_genre_english.csv",
        )
        df_original = df_original.rename(columns={
            "Testament (OT or NT)": "testamento",
            "Book Name": "libro",
            "Genre name": "genero",
            "Chapter": "capitulo",
            "Verse": "versiculo",
            "Text": "texto_original",
        })
        columnas = ["testamento", "libro", "genero", "capitulo", "versiculo", "texto_original"]
        self.df = df_original[columnas].reset_index(drop=True)

        with open(carpeta / "stopwords.json") as archivo:
            stopwords = set(json.load(archivo))
        self.preprocessor = TextPreprocessor(stopwords=stopwords)

        print("[servicio] Preprocesando versiculos...")
        textos = self.df["texto_original"].tolist()

        self.df["tokens"] = self.preprocessor.process_corpus(textos)
        self.tokens_para_ngramas = self.preprocessor.process_corpus_ngram(textos)

        largos = []
        for texto in self.df["texto_original"]:
            largos.append(len(texto.split()))
        self.df["n_palabras"] = largos

        self._construir_tfidf()
        self._construir_word2vec()
        self._construir_ngramas()
        print("[servicio] Listo.")

    #  CONSTRUCCION DE MODELOS
    def _construir_tfidf(self):
        print(f"[servicio] Construyendo indice TF-IDF (maximo {MAX_TERMINOS} terminos)...")

        conteo_palabras = {}
        for tokens in self.df["tokens"]:
            for palabra in tokens:
                if palabra in conteo_palabras:
                    conteo_palabras[palabra] += 1
                else:
                    conteo_palabras[palabra] = 1

        palabras_ordenadas = sorted(conteo_palabras.items(), key=lambda par: par[1], reverse=True)
        self.vocabulario_top = set()
        for palabra, _ in palabras_ordenadas[:MAX_TERMINOS]:
            self.vocabulario_top.add(palabra)

        documentos = []
        for tokens in self.df["tokens"]:
            filtrado = []
            for palabra in tokens:
                if palabra in self.vocabulario_top:
                    filtrado.append(palabra)
            documentos.append(filtrado)

        self.vectorizer = TFIDFVectorizer(normalizar=True)
        matriz = self.vectorizer.fit_transform(documentos)
        self.matriz_tfidf = matriz.astype(np.float32)

    def _construir_word2vec(self):
        print("[servicio] Entrenando Word2Vec...")

        self.modelo_w2v = Word2Vec(
            sentences=self.df["tokens"].tolist(),
            vector_size=DIM_W2V,
            window=5,       
            min_count=2,     
            workers=4,
            epochs=10,
            seed=42,
        )

        vectores = self.modelo_w2v.wv
        matriz_versiculos = np.zeros((len(self.df), DIM_W2V), dtype=np.float32)
        for i, tokens in enumerate(self.df["tokens"]):
            vectores_palabra = []
            for palabra in tokens:
                if palabra in vectores:
                    vectores_palabra.append(vectores[palabra])
            if len(vectores_palabra) > 0:
                matriz_versiculos[i] = np.mean(vectores_palabra, axis=0)
        self.matriz_w2v = matriz_versiculos

    def _construir_ngramas(self):
        print("[servicio] Entrenando modelos de n-gramas (n=1 a 5)...")

        self.modelos_ngrama = {}
        for n in range(1, 6):
            modelo = NGramModel(n)
            modelo.fit(self.tokens_para_ngramas)
            self.modelos_ngrama[n] = modelo

    def _filtrar(self, testamento=None, libro=None, capitulo=None):
        df = self.df
        if testamento:
            df = df[df["testamento"] == testamento]
        if libro:
            df = df[df["libro"] == libro]
        if capitulo is not None:
            df = df[df["capitulo"] == int(capitulo)]
        return df

    #  METADATOS
    def meta(self):
        libros = []
        ya_vistos = set()
        for _, fila in self.df.iterrows():
            nombre = fila["libro"]
            if nombre not in ya_vistos:
                ya_vistos.add(nombre)
                libros.append({"libro": nombre, "testamento": fila["testamento"]})
        libros.sort(key=lambda d: d["libro"])

        capitulos_por_libro = {}
        for _, fila in self.df.iterrows():
            nombre = fila["libro"]
            cap = int(fila["capitulo"])
            if nombre not in capitulos_por_libro or cap > capitulos_por_libro[nombre]:
                capitulos_por_libro[nombre] = cap

        testamentos = sorted(self.df["testamento"].unique().tolist())
        return {
            "testamentos": testamentos,
            "libros": libros,
            "capitulos_por_libro": capitulos_por_libro,
            "n_versiculos": int(len(self.df)),
            "n_libros": int(self.df["libro"].nunique()),
            "vocabulario": int(len(self.vectorizer.vocabulario)),
        }

    #  DASHBOARD
    def versiculos_por_libro(self, **filtros):
        df = self._filtrar(**filtros)
        conteo = {}
        for libro in df["libro"]:
            conteo[libro] = conteo.get(libro, 0) + 1
        resultado = []
        for libro, cantidad in conteo.items():
            resultado.append({"libro": libro, "n_versiculos": cantidad})
        resultado.sort(key=lambda d: d["n_versiculos"], reverse=True)
        return resultado

    def longitud_promedio(self, **filtros):
        df = self._filtrar(**filtros)
        suma = {}
        cantidad = {}
        for libro, n_palabras in zip(df["libro"], df["n_palabras"]):
            suma[libro] = suma.get(libro, 0) + n_palabras
            cantidad[libro] = cantidad.get(libro, 0) + 1
        resultado = []
        for libro in suma:
            promedio = suma[libro] / cantidad[libro]
            resultado.append({"libro": libro, "longitud_promedio": round(promedio, 2)})
        resultado.sort(key=lambda d: d["longitud_promedio"], reverse=True)
        return resultado

    def top_palabras(self, n=20, **filtros):
        df = self._filtrar(**filtros)
        conteo = {}
        for tokens in df["tokens"]:
            for palabra in tokens:
                conteo[palabra] = conteo.get(palabra, 0) + 1
        ordenadas = sorted(conteo.items(), key=lambda par: par[1], reverse=True)
        resultado = []
        for palabra, frecuencia in ordenadas[:n]:
            resultado.append({"palabra": palabra, "frecuencia": frecuencia})
        return resultado

    #  BUSCADOR SEMANTICO
    def buscar(self, query, k=10):
        tokens = self.preprocessor.process(query)
        vector_consulta = self.vectorizer.vectorizar_texto_nuevo(tokens).astype(np.float32)

        if not np.any(vector_consulta):
            return []

        similitudes = self.matriz_tfidf.dot(vector_consulta)

        pares = []
        for i in range(len(similitudes)):
            pares.append((i, similitudes[i]))
        pares.sort(key=lambda p: p[1], reverse=True)
        resultados = []
        for i, similitud in pares[:k]:
            fila = self.df.iloc[i]
            resultados.append({
                "libro": fila["libro"],
                "capitulo": int(fila["capitulo"]),
                "versiculo": int(fila["versiculo"]),
                "texto": fila["texto_original"],
                "similitud": round(float(similitud), 4),
            })
        return resultados

    #  PROYECCIONES
    def _muestra_por_libro(self, cap_por_libro=40, semilla=42):
        indices = []
        for libro in self.df["libro"].unique():
            posiciones = self.df.index[self.df["libro"] == libro]
            cuantos = min(len(posiciones), cap_por_libro)
            elegidos = pd.Series(posiciones).sample(cuantos, random_state=semilla)
            for pos in elegidos:
                indices.append(pos)
        return indices

    def _proyectar(self, matriz, indices, dim):
        datos = matriz[indices]

        pca = PCA(n_components=dim, random_state=42)
        coordenadas = pca.fit_transform(datos)

        varianza = []
        for valor in pca.explained_variance_ratio_:
            varianza.append(round(float(valor) * 100, 2))

        puntos = []
        for posicion, indice in enumerate(indices):
            fila = self.df.iloc[indice]
            punto = {
                "x": round(float(coordenadas[posicion][0]), 4),
                "y": round(float(coordenadas[posicion][1]), 4),
                "testamento": fila["testamento"],
                "genero": fila["genero"],
                "libro": fila["libro"],
                "ref": f'{fila["libro"]} {int(fila["capitulo"])}:{int(fila["versiculo"])}',
                "texto": fila["texto_original"][:90],
            }
            if dim == 3:
                punto["z"] = round(float(coordenadas[posicion][2]), 4)
            puntos.append(punto)

        return {"puntos": puntos, "varianza_explicada": varianza}

    def proyeccion_pca(self, dim=2, cap_por_libro=40):
        indices = self._muestra_por_libro(cap_por_libro)
        return self._proyectar(self.matriz_tfidf, indices, dim)

    def proyeccion_word2vec(self, dim=2, cap_por_libro=40):
        indices = self._muestra_por_libro(cap_por_libro)
        return self._proyectar(self.matriz_w2v, indices, dim)

    #  GENERADOR DE VERSICULOS
    def generar(self, modelo="bigram", palabra_inicial=None, max_len=20):
        nombres = {"unigram": 1, "bigram": 2, "trigram": 3, "tetragram": 4, "pentagram": 5}
        if str(modelo).lower() in nombres:
            n = nombres[str(modelo).lower()]
        else:
            try:
                n = int(modelo)
            except (TypeError, ValueError):
                n = 2
        if n < 1:
            n = 1
        if n > 5:
            n = 5

        palabra = palabra_inicial if palabra_inicial else None
        texto = self.modelos_ngrama[n].generar(palabra_inicial=palabra, max_len=int(max_len))
        return {"modelo": modelo, "n": n, "texto": texto}