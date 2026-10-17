"""
Utilidades compartidas por las paginas de la app (Streamlit): rutas, paleta de
colores, carga de datos y de resultados de los modelos.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]

RUTA_SPLIT = RAIZ / "data" / "splits" / "split.csv"
RUTA_IMAGENES = RAIZ / "data" / "processed" / "images_224"
RUTA_EMBEDDINGS = RAIZ / "data" / "embeddings"
RUTA_RESULTADOS = RAIZ / "resultados"
RUTA_MODELOS = RAIZ / "models"
RUTA_FINETUNING = RUTA_MODELOS / "finetuning"

# ----------------------------------------------------------------- paleta
# Okabe-Ito: paleta pensada para que se distinga con daltonismo. Cada clase
# tiene el mismo color en todos los graficos de la app.
CLASES = ["Bueno", "Maleza", "Otro daño", "Sequía"]
COLOR_CLASE = {
    "Bueno": "#009E73",
    "Maleza": "#0072B2",
    "Otro daño": "#CC79A7",
    "Sequía": "#E69F00",
}
COLOR_TIPO = {"Clásico sobre embeddings": "#8C96A0", "Fine-tuning": "#2E3A46"}
COLOR_AZAR = "#7A7A7A"

NOMBRES_DANO = {
    "DR": "Sequía", "DS": "Enfermedad", "FD": "Inundación", "G": "Bueno",
    "ND": "Deficiencia de nutrientes", "PS": "Plaga", "WD": "Maleza", "WN": "Viento",
}
NOMBRES_ETAPA = {"S": "Siembra", "V": "Vegetativo", "F": "Floración", "M": "Madurez"}
ORDEN_ETAPAS = ["Siembra", "Vegetativo", "Floración", "Madurez"]
ORDEN_TEMPORADAS = ["LR2020", "SR2020", "LR2021", "SR2021"]

NOMBRES_EXTRACTOR = {"resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0"}
NOMBRES_ALGORITMO = {
    "regresion_logistica": "Regresión logística",
    "svm": "SVM",
    "random_forest": "Random Forest",
}


def color_clase_de_dano(dano):
    """Color de una de las 8 clases originales: las cinco minoritarias comparten el de 'Otro daño'."""
    return COLOR_CLASE.get(dano, COLOR_CLASE["Otro daño"])


def chip(texto, color):
    """Etiqueta de color en HTML (para mostrar la clase predicha)."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    luminancia = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    texto_color = "#1F2328" if luminancia > 0.6 else "#FFFFFF"
    return (
        f'<span style="background:{color};color:{texto_color};padding:3px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.9em;">{texto}</span>'
    )


# ------------------------------------------------------------------ datos

@st.cache_data(show_spinner=False)
def cargar_datos():
    """split.csv con nombres legibles de cada variable."""
    if not RUTA_SPLIT.exists():
        return None
    df = pd.read_csv(RUTA_SPLIT)
    df["daño"] = df["damage"].map(NOMBRES_DANO)
    df["etapa"] = df["growth_stage"].map(NOMBRES_ETAPA)
    df["temporada"] = df["season"]
    df["archivo_224"] = df["filename"].map(lambda f: Path(f).stem + ".jpg")
    return df


def exigir_datos():
    """Devuelve los datos o detiene la pagina con un mensaje claro."""
    df = cargar_datos()
    if df is None:
        st.error(f"No se encontró {RUTA_SPLIT}. Corré primero el notebook 02 (partición de datos).")
        st.stop()
    return df


@st.cache_data(show_spinner=False)
def archivos_disponibles():
    """Nombres de las fotos redimensionadas que hay en disco (se lista la carpeta una sola vez)."""
    if not RUTA_IMAGENES.exists():
        return frozenset()
    return frozenset(p.name for p in RUTA_IMAGENES.glob("*.jpg"))


def ruta_foto(archivo_224):
    """Ruta de una foto redimensionada, o None si no esta en disco."""
    ruta = RUTA_IMAGENES / archivo_224
    return ruta if ruta.exists() else None


# ------------------------------------------------------------- resultados

@st.cache_data(show_spinner=False)
def cargar_resultados(objetivo="clase4"):
    """
    Tabla unificada de resultados en test: modelos clasicos (notebook 05) y, para
    4 clases, los fine-tuneados (notebook 06). Devuelve (DataFrame, referencias).
    """
    ruta = RUTA_RESULTADOS / f"clasicos_{objetivo}.csv"
    if not ruta.exists():
        return None, {}

    clasicos = pd.read_csv(ruta)
    referencias = {
        fila["modelo"]: fila
        for _, fila in clasicos[clasicos["embedding"] == "-"].iterrows()
    }
    tabla = clasicos[clasicos["embedding"] != "-"].copy()
    tabla["tipo"] = "Clásico sobre embeddings"
    tabla["etiqueta"] = (
        tabla["modelo"].map(NOMBRES_ALGORITMO) + " + " + tabla["embedding"].map(NOMBRES_EXTRACTOR)
    )
    tabla["id"] = "cl:" + tabla["embedding"] + ":" + tabla["modelo"]

    if objetivo == "clase4":
        filas = []
        for nombre in NOMBRES_EXTRACTOR:
            ruta_m = RUTA_FINETUNING / nombre / "metricas.json"
            if not ruta_m.exists():
                continue
            m = json.loads(ruta_m.read_text(encoding="utf-8"))
            fila = {
                "embedding": "(fine-tuning)", "modelo": nombre,
                "parametros": f"mejor época {m['mejor_epoca']} de {m['epocas_corridas']}",
                "f1_macro_val": m["f1_macro_val"], "f1_macro_test": m["test"]["f1_macro"],
                "balanced_accuracy_test": m["test"]["balanced_accuracy"],
                "accuracy_test": m["test"]["accuracy"],
                "tiempo_entrenamiento_s": m["tiempo_entrenamiento_s"],
                "tiempo_prediccion_s": m["tiempo_prediccion_test_s"],
                "tipo": "Fine-tuning",
                "etiqueta": f"{NOMBRES_EXTRACTOR[nombre]} (fine-tuning)",
                "id": f"ft:{nombre}",
            }
            fila.update({f"f1_{c}": v for c, v in m["test"]["f1_por_clase"].items()})
            filas.append(fila)
        if filas:
            tabla = pd.concat([tabla, pd.DataFrame(filas)], ignore_index=True)

    return tabla, referencias


@st.cache_data(show_spinner=False)
def cargar_historial(nombre):
    ruta = RUTA_FINETUNING / nombre / "historial.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data(show_spinner=False)
def cargar_predicciones_finetuning(nombre):
    """(clases reales, clases predichas) sobre test, guardadas en el notebook 06."""
    base = RUTA_FINETUNING / nombre
    if not (base / "real_test.npy").exists():
        return None
    clases = json.loads((RUTA_FINETUNING / "clases.json").read_text(encoding="utf-8")) \
        if (RUTA_FINETUNING / "clases.json").exists() else CLASES
    real = np.array(clases)[np.load(base / "real_test.npy")]
    pred = np.array(clases)[np.load(base / "pred_test.npy")]
    return real, pred
