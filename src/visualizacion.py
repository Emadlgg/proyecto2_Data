"""
Funciones de graficado reutilizables para el analisis exploratorio.
Cada funcion guarda automaticamente la figura en figuras/ si se le
pasa un nombre de archivo.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def _guardar(fig, nombre_archivo, carpeta="figuras"):
    if nombre_archivo:
        os.makedirs(carpeta, exist_ok=True)
        fig.savefig(os.path.join(carpeta, nombre_archivo), dpi=150, bbox_inches="tight")


def histograma(df, columna, titulo=None, bins=20, nombre_archivo=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[columna], bins=bins, ax=ax)
    ax.set_title(titulo or f"Distribucion de {columna}")
    _guardar(fig, nombre_archivo)
    return fig


def boxplot_por_grupo(df, columna_valor, columna_grupo, titulo=None, nombre_archivo=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x=columna_grupo, y=columna_valor, ax=ax)
    ax.set_title(titulo or f"{columna_valor} por {columna_grupo}")
    plt.xticks(rotation=30, ha="right")
    _guardar(fig, nombre_archivo)
    return fig


def barras_frecuencia(df, columna, titulo=None, nombre_archivo=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    orden = df[columna].value_counts().index
    sns.countplot(data=df, x=columna, order=orden, ax=ax)
    ax.set_title(titulo or f"Frecuencia de {columna}")
    plt.xticks(rotation=30, ha="right")
    _guardar(fig, nombre_archivo)
    return fig


def tabla_cruzada(df, fila, columna, normalizar=False):
    """Devuelve una tabla de contingencia (frecuencias o proporciones)."""
    return pd.crosstab(df[fila], df[columna], normalize="index" if normalizar else False)


def grilla_imagenes(rutas_imagenes, etiquetas=None, cols=5, titulo=None, nombre_archivo=None):
    """Muestra una grilla de imagenes de muestra, una por celda."""
    from PIL import Image

    n = len(rutas_imagenes)
    filas = (n + cols - 1) // cols
    fig, axes = plt.subplots(filas, cols, figsize=(cols * 2.5, filas * 2.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            img = Image.open(rutas_imagenes[i])
            ax.imshow(img)
            if etiquetas is not None:
                ax.set_title(str(etiquetas[i]), fontsize=9)
        ax.axis("off")

    if titulo:
        fig.suptitle(titulo)
    _guardar(fig, nombre_archivo)
    return fig
