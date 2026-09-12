"""
Funciones de limpieza y preparacion de datos para el analisis exploratorio
del reto CGIAR Eyes on the Ground.
"""

import pandas as pd

DAMAGE_LABELS = {
    "DR": "Sequia",
    "DS": "Enfermedad",
    "FD": "Inundacion",
    "G": "Bueno",
    "ND": "Deficiencia de nutrientes",
    "PS": "Plaga",
    "WD": "Maleza",
    "WN": "Viento",
}

GROWTH_STAGE_LABELS = {
    "S": "Siembra",
    "V": "Vegetativo",
    "F": "Floracion",
    "M": "Madurez",
}


def cargar_train(ruta="data/Train.csv"):
    """Carga Train.csv y agrega columnas legibles de damage y growth_stage."""
    df = pd.read_csv(ruta)
    df["damage_label"] = df["damage"].map(DAMAGE_LABELS)
    df["growth_stage_label"] = df["growth_stage"].map(GROWTH_STAGE_LABELS)
    return df


def resumen_nulos(df):
    """Tabla rapida de nulos por columna, solo las que tienen al menos uno."""
    nulos = df.isnull().sum()
    return nulos[nulos > 0]


def filtrar_sequia(df):
    """
    Devuelve solo las filas donde damage == 'DR', que es el unico tipo de
    dano con valores de extent distintos de 0. Util para el EDA cuantitativo
    de severidad.
    """
    return df[df["damage"] == "DR"].copy()


def muestra_balanceada_por_clase(df, columna="damage", n_por_clase=15, random_state=42):
    """
    Devuelve una muestra con hasta n_por_clase filas de cada valor de
    `columna`, util para elegir que imagenes extraer del zip.
    """
    return (
        df.groupby(columna, group_keys=False)
        .apply(lambda g: g.sample(min(n_por_clase, len(g)), random_state=random_state))
        .reset_index(drop=True)
    )
