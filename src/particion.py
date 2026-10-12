"""
Particion de datos para el modelado (Proyecto 2 - CGIAR Eyes on the Ground).

La particion es agrupada: las fotos que comparten identificador de campo
(prefijo del nombre de archivo) quedan siempre en el mismo conjunto, para
que el modelo no pueda "reconocer el campo" en vez del dano.
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CLASE4_POR_DAMAGE = {"G": "Bueno", "WD": "Maleza", "DR": "Sequía"}
OTRO_DANO = "Otro daño"


def mapear_clase4(damage):
    """Agrupa las 8 clases originales en 4: Bueno, Maleza, Sequía y Otro daño."""
    return damage.map(CLASE4_POR_DAMAGE).fillna(OTRO_DANO)


def asignar_grupos(df):
    """
    Devuelve una Serie con el grupo de cada foto.

    - Fotos tipo 'L427F01330C01S03961Rp02052.jpg': grupo = prefijo 'L427F01330'.
    - Fotos tipo '24_initial_1_1463_1463.JPG' (LR2020): grupo = 'N24'.
    - Cualquier otra: su propio nombre de archivo (grupo de una sola foto).
    """
    prefijo = df["filename"].str.extract(r"^(L\d+F\d+)")[0]
    numero = df["filename"].str.extract(r"^(\d+)_")[0]
    grupo = prefijo.fillna("N" + numero)
    return grupo.fillna(df["filename"])


def particionar(df, semilla=42, col_estratificar="damage"):
    """
    Agrega la columna 'split' (train / val / test) con una particion agrupada
    y estratificada, de aprox. 71% / 14% / 14%.

    Se estratifica por las 8 clases originales para que las clases raras
    queden repartidas y la misma particion sirva para el experimento de
    4 clases y para el de 8.
    """
    df = df.copy()

    # 1) separar test (~1/7 de los datos)
    cv_test = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=semilla)
    idx_tv, idx_test = next(cv_test.split(df, df[col_estratificar], df["grupo"]))
    tv = df.iloc[idx_tv]

    # 2) del resto, separar validacion (~1/6 del resto)
    cv_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=semilla)
    idx_train, idx_val = next(cv_val.split(tv, tv[col_estratificar], tv["grupo"]))

    df["split"] = "test"
    df.loc[tv.index[idx_train], "split"] = "train"
    df.loc[tv.index[idx_val], "split"] = "val"
    return df


def verificar_particion(df):
    """Comprueba que ningun grupo aparezca en mas de un conjunto."""
    n_conjuntos = df.groupby("grupo")["split"].nunique()
    solapados = n_conjuntos[n_conjuntos > 1]
    assert solapados.empty, f"{len(solapados)} grupos aparecen en mas de un conjunto"
    return True
