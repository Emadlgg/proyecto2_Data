"""
Parte de la app que prepara la foto y la clasifica con los modelos entrenados.

torch se importa solo cuando se clasifica una foto, asi las demas paginas de la
app funcionan aunque torch no este instalado.
"""

import io
import json
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src.app_comun import (
    CLASES, NOMBRES_ALGORITMO, NOMBRES_DANO, NOMBRES_EXTRACTOR, RUTA_EMBEDDINGS, RUTA_FINETUNING,
    RUTA_MODELOS,
)

TAMANO = 224


class ErrorDeFoto(Exception):
    """El archivo no se pudo leer como imagen."""


class DependenciaFaltante(Exception):
    """Falta una libreria necesaria para clasificar."""


# ------------------------------------------------------- preprocesamiento

def preparar_imagen(origen):
    """
    Deja una foto lista para los modelos, igual que en el entrenamiento
    (notebook 03): corrige la orientacion EXIF, pasa a RGB, redimensiona a
    224x224 y la guarda como JPEG de calidad 95, que es como se guardaron las
    fotos de entrenamiento.

    `origen` puede ser una ruta, bytes o un archivo subido en Streamlit.
    """
    try:
        if isinstance(origen, (bytes, bytearray)):
            origen = io.BytesIO(origen)
        im = Image.open(origen)
        if im.format == "JPEG":
            im.draft("RGB", (TAMANO * 2, TAMANO * 2))  # acelera fotos grandes
        im = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ErrorDeFoto("El archivo no se pudo leer como imagen (puede estar dañado o no ser una foto).") from e

    im = im.resize((TAMANO, TAMANO), Image.LANCZOS)
    buffer = io.BytesIO()
    im.save(buffer, "JPEG", quality=95)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# ------------------------------------------------------ modelos disponibles

def opciones_modelos():
    """
    {id: info} con los modelos que hay en disco. Los ids son 'ft:<modelo>' para
    los fine-tuneados y 'cl:<extractor>:<algoritmo>' para los clasicos.
    """
    opciones = {}
    for nombre, etiqueta in NOMBRES_EXTRACTOR.items():
        archivo = RUTA_FINETUNING / nombre / "mejor.pt"
        opciones[f"ft:{nombre}"] = {
            "nombre": f"{etiqueta} (fine-tuning)", "tipo": "Fine-tuning",
            "archivo": archivo, "disponible": archivo.exists(),
        }
    for algoritmo, nombre_alg in NOMBRES_ALGORITMO.items():
        for extractor, nombre_ext in NOMBRES_EXTRACTOR.items():
            archivo = RUTA_MODELOS / f"{extractor}_{algoritmo}_clase4.joblib"
            opciones[f"cl:{extractor}:{algoritmo}"] = {
                "nombre": f"{nombre_alg} + {nombre_ext}", "tipo": "Clásico sobre embeddings",
                "archivo": archivo, "disponible": archivo.exists(),
            }
    return opciones


def clases_finetuning():
    ruta = RUTA_FINETUNING / "clases.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else CLASES


# --------------------------------------------------------- partes puras

def softmax(logits):
    z = np.asarray(logits, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def clasificar_con_pipeline(pipe, vector):
    """
    Clase predicha por un modelo clasico y, si el modelo las da, las
    probabilidades por clase (el SVM entrenado no las entrega: devuelve None).
    """
    clase = str(pipe.predict(vector)[0])
    probas = None
    if hasattr(pipe, "predict_proba"):
        valores = pipe.predict_proba(vector)[0]
        probas = {str(c): float(p) for c, p in zip(pipe.classes_, valores)}
    return clase, probas


# ------------------------------------------------------------- con torch

def _importar_torch():
    try:
        import torch
    except ImportError as e:
        raise DependenciaFaltante(
            "Falta instalar torch y torchvision para clasificar fotos (pip install torch torchvision)."
        ) from e
    return torch


@st.cache_resource(show_spinner="Cargando el modelo fine-tuneado...")
def _cargar_finetuned(nombre):
    torch = _importar_torch()
    from src.finetuning import crear_modelo

    ruta = RUTA_FINETUNING / nombre / "mejor.pt"
    modelo = crear_modelo(nombre, len(clases_finetuning()), pesos_imagenet=False)
    modelo.load_state_dict(torch.load(ruta, map_location="cpu"))
    return modelo.eval()


@st.cache_resource(show_spinner="Cargando el extractor de características (la primera vez descarga los pesos de ImageNet)...")
def _cargar_extractor(nombre):
    _importar_torch()
    from src.embeddings import crear_extractor

    modelo, _ = crear_extractor(nombre, pesos=True)
    return modelo


@st.cache_resource(show_spinner="Cargando el modelo...")
def _cargar_clasico(extractor, algoritmo):
    return joblib.load(RUTA_MODELOS / f"{extractor}_{algoritmo}_clase4.joblib")


def clasificar(imagen, ids):
    """
    Clasifica una foto ya preparada (224x224) con los modelos indicados.
    Devuelve una lista de dicts: id, clase, probas (o None), segundos, error.
    """
    torch = _importar_torch()
    from src.embeddings import TRANSFORMACION

    tensor = TRANSFORMACION(imagen).unsqueeze(0)
    opciones = opciones_modelos()
    embeddings = {}  # un embedding por extractor, para no repetirlo en cada modelo clasico
    resultados = []

    for id_ in ids:
        t0 = time.time()
        fila = {"id": id_, "clase": None, "probas": None, "segundos": None, "error": None}
        try:
            if not opciones[id_]["disponible"]:
                raise FileNotFoundError(f"No se encontró {opciones[id_]['archivo']}")

            partes = id_.split(":")
            if partes[0] == "ft":
                modelo = _cargar_finetuned(partes[1])
                with torch.no_grad():
                    logits = modelo(tensor)[0].numpy()
                probas = softmax(logits)
                clases = clases_finetuning()
                fila["clase"] = clases[int(probas.argmax())]
                fila["probas"] = {c: float(p) for c, p in zip(clases, probas)}
            else:
                _, extractor, algoritmo = partes
                if extractor not in embeddings:
                    with torch.no_grad():
                        embeddings[extractor] = _cargar_extractor(extractor)(tensor).numpy()
                pipe = _cargar_clasico(extractor, algoritmo)
                fila["clase"], fila["probas"] = clasificar_con_pipeline(pipe, embeddings[extractor])
        except Exception as e:  # un modelo que falla no debe tumbar a los demas
            fila["error"] = f"{type(e).__name__}: {e}"
        fila["segundos"] = time.time() - t0
        resultados.append(fila)
    return resultados


# --------------------------------------- predicciones de test (sin torch)

@st.cache_data(show_spinner="Calculando las predicciones del modelo sobre test...")
def predicciones_clasico_test(extractor, algoritmo, objetivo="clase4"):
    """
    (clases reales, clases predichas) sobre test, usando los embeddings guardados.
    objetivo: "clase4" (4 clases) o "damage" (las 8 clases originales).
    """
    indice = pd.read_csv(RUTA_EMBEDDINGS / "indice.csv")
    X = np.load(RUTA_EMBEDDINGS / f"{extractor}.npy")
    es_test = (indice["split"] == "test").values
    pipe = joblib.load(RUTA_MODELOS / f"{extractor}_{algoritmo}_{objetivo}.joblib")
    if objetivo == "damage":
        reales = indice.loc[es_test, "damage"].map(NOMBRES_DANO).values
    else:
        reales = indice.loc[es_test, "clase4"].values
    return reales, pipe.predict(X[es_test])
