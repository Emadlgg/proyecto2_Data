"""
Redimensiona las fotos del dataset directo desde el zip, sin descomprimir todo.

Las funciones de trabajo viven en este modulo (y no en el notebook) para que
funcionen con multiprocessing en Windows.
"""

import io
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

TAMANO = 224
CALIDAD = 95


def _indice_zip(zf):
    """Devuelve {nombre_de_archivo: ruta_dentro_del_zip}, sin importar subcarpetas."""
    return {os.path.basename(n): n for n in zf.namelist() if not n.endswith("/")}


def _procesar_imagen(zf, miembro, ruta_salida, tamano, calidad):
    """Lee una foto del zip, la redimensiona a tamano x tamano y la guarda como JPEG."""
    with zf.open(miembro) as f:
        datos = f.read()

    im = Image.open(io.BytesIO(datos))
    ancho, alto = im.size
    orientacion = im.getexif().get(274, 1)  # 274 = etiqueta EXIF de orientacion
    if orientacion in (5, 6, 7, 8):  # la foto esta girada: ancho y alto se intercambian
        ancho, alto = alto, ancho

    im.draft("RGB", (tamano * 2, tamano * 2))  # acelera la lectura de JPEG grandes
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB").resize((tamano, tamano), Image.LANCZOS)
    im.save(ruta_salida, "JPEG", quality=calidad)
    return ancho, alto, orientacion


def procesar_lote(args):
    """Procesa una lista de fotos. Cada trabajador abre su propio ZipFile."""
    ruta_zip, nombres, destino, tamano, calidad = args
    filas = []
    with zipfile.ZipFile(ruta_zip, "r") as zf:
        indice = _indice_zip(zf)
        for nombre in nombres:
            salida = f"{Path(nombre).stem}.jpg"
            fila = {
                "filename": nombre,
                "archivo_224": salida,
                "ancho_orig": None,
                "alto_orig": None,
                "orientacion_exif": None,
                "ok": False,
                "error": "",
            }
            if nombre not in indice:
                fila["error"] = "no esta en el zip"
            else:
                try:
                    ancho, alto, orient = _procesar_imagen(
                        zf, indice[nombre], os.path.join(destino, salida), tamano, calidad
                    )
                    fila.update(ancho_orig=ancho, alto_orig=alto, orientacion_exif=orient, ok=True)
                except Exception as e:  # archivo corrupto, truncado o ilegible
                    fila["error"] = f"{type(e).__name__}: {e}"[:200]
            filas.append(fila)
    return pd.DataFrame(filas)


def redimensionar_desde_zip(
    ruta_zip,
    nombres,
    destino,
    tamano=TAMANO,
    calidad=CALIDAD,
    tam_lote=500,
    workers=None,
):
    """
    Redimensiona todas las fotos de `nombres` desde el zip a `destino`.

    Guarda un checkpoint por lote en destino/_lotes/, asi que si se interrumpe
    se puede volver a correr y continua donde se quedo.

    Returns
    -------
    pandas.DataFrame con una fila por foto (ok, error, dimensiones originales).
    """
    destino = Path(destino)
    carpeta_lotes = destino / "_lotes"
    carpeta_lotes.mkdir(parents=True, exist_ok=True)

    nombres = list(nombres)
    lotes = [nombres[i : i + tam_lote] for i in range(0, len(nombres), tam_lote)]
    resultados = {}
    pendientes = []

    for i, lote in enumerate(lotes):
        ckpt = carpeta_lotes / f"lote_{i:04d}.csv"
        if ckpt.exists():
            resultados[i] = pd.read_csv(ckpt, keep_default_na=False)
        else:
            pendientes.append((i, lote, ckpt))

    print(f"Lotes totales: {len(lotes)} | ya hechos: {len(resultados)} | pendientes: {len(pendientes)}")

    if pendientes:
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futuros = {
                pool.submit(procesar_lote, (str(ruta_zip), lote, str(destino), tamano, calidad)): (i, ckpt)
                for i, lote, ckpt in pendientes
            }
            for n, fut in enumerate(as_completed(futuros), 1):
                i, ckpt = futuros[fut]
                df_lote = fut.result()
                df_lote.to_csv(ckpt, index=False)
                resultados[i] = df_lote
                if n % 5 == 0 or n == len(pendientes):
                    print(f"  {n}/{len(pendientes)} lotes procesados")

    meta = pd.concat([resultados[i] for i in sorted(resultados)], ignore_index=True)
    meta["ok"] = meta["ok"].astype(str).str.lower().eq("true")
    for col in ("ancho_orig", "alto_orig", "orientacion_exif"):
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta["error"] = meta["error"].fillna("").astype(str)
    return meta
