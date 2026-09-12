"""
Descarga los zips de imagenes (train/test) desde Google Drive solo si no
existen localmente, y permite extraer una muestra puntual de imagenes sin
descomprimir el zip completo (los zips pesan varios GB).
"""

import os
import zipfile
import gdown


FILE_IDS = {
    "train": "13UgIaIUhbvskcJ3fstlZcqn115Svgq_P",
    "test": "1-AwdqlpdzUaMtDZK5RfzX6arRrWiy_Ad",
}


def descargar_zip_si_no_existe(split, destino_dir="data/raw"):
    """
    Descarga train.zip o test.zip desde Drive si todavia no existe localmente.

    Parameters
    ----------
    split : str
        "train" o "test"
    destino_dir : str
        Carpeta donde se guarda el zip descargado

    Returns
    -------
    str
        Ruta local del zip
    """
    if split not in FILE_IDS:
        raise ValueError(f"split debe ser 'train' o 'test', recibido: {split}")

    ruta_zip = os.path.join(destino_dir, f"{split}.zip")
    if os.path.exists(ruta_zip):
        print(f"{split}.zip ya existe, no se vuelve a descargar.")
        return ruta_zip

    os.makedirs(destino_dir, exist_ok=True)
    url = f"https://drive.google.com/uc?id={FILE_IDS[split]}"
    gdown.download(url, ruta_zip, quiet=False)
    return ruta_zip


def extraer_muestra(nombres_archivo, split, destino_dir="data/raw"):
    """
    Extrae solo las imagenes indicadas desde el zip, sin descomprimir todo.

    Parameters
    ----------
    nombres_archivo : list[str]
        Lista de valores de la columna 'filename' del csv a extraer
    split : str
        "train" o "test"
    destino_dir : str
        Carpeta base donde vive el zip y donde se crea la carpeta de muestra

    Returns
    -------
    str
        Ruta de la carpeta con las imagenes extraidas
    """
    ruta_zip = os.path.join(destino_dir, f"{split}.zip")
    destino = os.path.join(destino_dir, f"sample_images_{split}")
    os.makedirs(destino, exist_ok=True)

    faltantes = [n for n in nombres_archivo if not os.path.exists(os.path.join(destino, n))]
    if not faltantes:
        print(f"Todas las imagenes de muestra de {split} ya estan extraidas.")
        return destino

    if not os.path.exists(ruta_zip):
        raise FileNotFoundError(
            f"No se encontro {ruta_zip}. Corre descargar_zip_si_no_existe('{split}') primero."
        )

    with zipfile.ZipFile(ruta_zip, "r") as z:
        nombres_en_zip = set(z.namelist())
        extraidas = 0
        no_encontradas = []
        for nombre in faltantes:
            candidatos = [n for n in nombres_en_zip if n.endswith(nombre)]
            if candidatos:
                z.extract(candidatos[0], destino)
                extraidas += 1
            else:
                no_encontradas.append(nombre)

    print(f"Extraidas {extraidas} imagenes nuevas de {split}.")
    if no_encontradas:
        print(f"No se encontraron {len(no_encontradas)} imagenes dentro del zip: {no_encontradas[:5]}...")

    return destino
