# Proyecto 2 - CGIAR Eyes on the Ground

Clasificación de daño en fotos de cultivos de maíz tomadas por agricultores en Kenia, con énfasis en la **sequía**. El proyecto parte del reto [CGIAR Eyes on the Ground](https://zindi.africa/competitions/cgiar-eyes-on-the-ground-challenge) de Zindi, pensado para automatizar la verificación de reclamos de seguro agrícola basado en imágenes (*picture-based insurance*).

Incluye análisis exploratorio, modelos de clasificación (CNN con fine-tuning y modelos clásicos sobre embeddings) y una aplicación interactiva en Streamlit.

**Autor:** Osman Emanuel de León García
**Curso:** CC3084 - Data Science, Universidad del Valle de Guatemala
**Fecha:** Octubre de 2026

## Resumen de resultados

**Hallazgos del análisis exploratorio**

- Hay 26,068 fotos de entrenamiento y 8 tipos de daño muy desbalanceados: Bueno (44.6%), Maleza (35.4%) y Sequía (17.3%) concentran casi todo; las otras cinco clases suman menos del 3%.
- La sequía se concentra en la etapa de **Madurez**: reúne el 71.7% de los casos de sequía, y dentro de esa etapa casi la mitad de las fotos (48.6%) muestran sequía.
- La etapa de crecimiento se asocia mucho más con el tipo de daño que la temporada (V de Cramér de 0.324 contra 0.120).
- La frecuencia de sequía es estable entre temporadas, pero su severidad no: SR2020 tiene la media más alta (53.5%) y LR2021 la más baja (34.5%).

**Modelos (4 clases: Bueno, Maleza, Sequía y Otro daño; conjunto de prueba)**

| Modelo | F1 macro | Balanced accuracy | Accuracy | F1 Sequía |
|---|---|---|---|---|
| ResNet50 (fine-tuning) | 0.588 | 0.606 | 0.687 | 0.700 |
| EfficientNet-B0 (fine-tuning) | 0.563 | 0.610 | 0.664 | 0.699 |
| SVM + EfficientNet-B0 | 0.546 | 0.536 | 0.664 | 0.655 |
| Regresión logística + EfficientNet-B0 | 0.534 | 0.600 | 0.622 | 0.646 |
| Random Forest + EfficientNet-B0 | 0.522 | 0.519 | 0.637 | 0.649 |
| Random Forest + ResNet50 | 0.509 | 0.510 | 0.618 | 0.649 |
| SVM + ResNet50 | 0.501 | 0.536 | 0.589 | 0.633 |
| Regresión logística + ResNet50 | 0.500 | 0.552 | 0.585 | 0.630 |
| *Referencia: al azar según proporción de clases* | 0.250 | 0.250 | 0.355 | 0.183 |
| *Referencia: siempre "Bueno"* | 0.154 | 0.250 | 0.446 | 0.000 |

El fine-tuning mejora a los modelos clásicos, pero de forma moderada. "Otro daño" sigue siendo la clase más difícil (F1 entre 0.14 y 0.26), y Bueno y Maleza se confunden entre sí. La diferencia entre los dos modelos fine-tuneados se debe en buena parte a "Otro daño", que tiene solo 100 fotos de prueba, así que no conviene darle demasiado peso.

**Experimento secundario con las 8 clases originales.** Se probó solo con modelos clásicos. El mejor F1 macro fue 0.319 (regresión logística + EfficientNet-B0). Inundación, Viento y Enfermedad tienen 2, 5 y 18 fotos de prueba, por lo que sus métricas son muy ruidosas. Esto respalda la decisión de usar 4 clases como experimento principal.

## Decisiones de diseño

- **4 clases como experimento principal.** Las cinco clases minoritarias se agrupan en "Otro daño" porque tienen muy pocas fotos para entrenar o evaluar. Las 8 clases se evalúan como experimento secundario.
- **Partición agrupada.** Los nombres de archivo repiten un prefijo (por ejemplo `L427F01330`) entre varias fotos; se asume que identifica un campo o parcela. Las fotos con el mismo prefijo quedan siempre en el mismo conjunto (≈71% train, 14% val, 14% test), para que el modelo no aprenda a reconocer el campo en vez del daño. La partición está en `data/splits/split.csv`.
- **Métrica principal:** F1 macro, porque la exactitud engaña con clases tan desbalanceadas. Test se usa una sola vez, al final.
- **Desbalance de clases:** pesos de clase en todos los modelos (`class_weight="balanced"` en los clásicos y pesos en la función de pérdida en el fine-tuning).
- **Fine-tuning en dos fases:** primero solo la capa final y después toda la red con una tasa de aprendizaje más baja, con aumento de datos suave que no altera el color.

## Estructura del proyecto

```
Proyecto2-CGIAR/
├── .streamlit/
│   └── config.toml                Tema de la app
├── app/
│   ├── Inicio.py                  Página principal de la app
│   └── pages/
│       ├── 1_Explorar_los_datos.py
│       ├── 2_Clasificar_una_foto.py
│       └── 3_Rendimiento_de_los_modelos.py
├── data/
│   ├── Train.csv, Test.csv, SampleSubmission.csv
│   ├── splits/
│   │   └── split.csv              Partición train / val / test
│   ├── raw/                       Zips de imágenes originales (no se versiona)
│   ├── processed/                 Fotos de 224x224 y metadatos (no se versiona)
│   └── embeddings/                Embeddings de las CNN (no se versiona)
├── figuras/                       Gráficos exportados para el informe
├── informe/                       Informe final (LaTeX y PDF)
├── models/                        Modelos entrenados (no se versiona)
│   └── finetuning/<modelo>/       Pesos y resultados de cada CNN
├── notebooks/
│   ├── 02_preparacion.ipynb
│   ├── 03_preprocesamiento_imagenes.ipynb
│   ├── 04_embeddings.ipynb        Corre en Colab (GPU)
│   ├── 05_modelos_clasicos.ipynb
│   └── 06_finetuning.ipynb        Corre en Colab (GPU)
├── presentacion/                  Presentación final
├── resultados/                    Tablas de resultados en CSV
├── src/
│   ├── descarga_imagenes.py       Descarga los zips desde Drive y extrae muestras
│   ├── limpieza.py                Carga de datos y etiquetas
│   ├── visualizacion.py           Gráficos del análisis exploratorio
│   ├── particion.py               Clases agrupadas y partición agrupada
│   ├── procesamiento_imagenes.py  Redimensionado de fotos desde el zip
│   ├── embeddings.py              Extractores preentrenados y preprocesamiento
│   ├── modelos_clasicos.py        Regresión logística, SVM y Random Forest
│   ├── finetuning.py              Entrenamiento de las CNN
│   ├── app_comun.py               Datos, rutas y colores de la app
│   └── app_modelos.py             Preparación de fotos y clasificación en la app
├── notebook_eda.ipynb             Análisis exploratorio
├── requirements.txt
└── README.md
```

## Cómo reproducir el proyecto

### 1. Preparar el entorno

```bash
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Colocar `Train.csv`, `Test.csv` y `SampleSubmission.csv` en `data/` (se descargan desde la pestaña "Data" del reto en Zindi).

### 2. Ejecutar los notebooks en orden

| Notebook | Dónde | Qué hace | Genera |
|---|---|---|---|
| `notebook_eda.ipynb` | Laptop | Análisis exploratorio. La primera vez descarga `train.zip` (~7 GB) desde Drive con los IDs de `src/descarga_imagenes.py` y extrae una muestra de fotos. | `figuras/` |
| `02_preparacion.ipynb` | Laptop | Agrupa las clases, arma los grupos por prefijo y hace la partición. | `data/splits/split.csv` |
| `03_preprocesamiento_imagenes.ipynb` | Laptop | Redimensiona las fotos a 224×224 directo desde el zip, detecta archivos corruptos y empaqueta el resultado. | `data/processed/` (`images_224.zip`, `imagenes_meta.csv`) |
| `04_embeddings.ipynb` | **Colab (GPU)** | Extrae embeddings con ResNet50 y EfficientNet-B0. | `embeddings/` en Drive |
| `05_modelos_clasicos.ipynb` | Laptop | Entrena regresión logística, SVM y Random Forest sobre los embeddings. Con `OBJETIVO = "damage"` repite el experimento con 8 clases. | `resultados/`, `models/*.joblib` |
| `06_finetuning.ipynb` | **Colab (GPU)** | Fine-tuning de EfficientNet-B0 y ResNet50. | `finetuning/` en Drive |

**Notebooks de Colab (04 y 06):**

1. Subir a una carpeta de Drive (por defecto `Proyecto2-CGIAR`) los archivos `images_224.zip` e `imagenes_meta.csv` que salen del notebook 03.
2. Subir el código al repositorio de GitHub, porque los notebooks lo clonan para usar `src/`.
3. En la primera celda de cada notebook, poner `REPO_URL` y, si cambió, `CARPETA_DRIVE`. Si el repositorio es privado, la URL necesita un token.
4. Activar la GPU (*Entorno de ejecución → Cambiar tipo de entorno → GPU*).
5. Hacer primero una prueba rápida con `LIMITE = 300` y después correr la versión completa con `LIMITE = None`.

Después de correr el notebook 04, bajar `indice.csv`, `resnet50.npy` y `efficientnet_b0.npy` a `data/embeddings/`. Después del 06, bajar la carpeta `finetuning/` de Drive a `models/finetuning/` y copiar `finetuning_clase4.csv` a `resultados/`.

### 3. Correr la aplicación

Desde la raíz del proyecto:

```bash
streamlit run app/Inicio.py
```

La aplicación tiene cuatro páginas:

- **Inicio:** el problema y el hallazgo principal.
- **Explorar los datos:** filtros por temporada, etapa y tipo de daño, con gráficos interactivos y fotos de ejemplo.
- **Clasificar una foto:** se sube una foto (o se elige una del conjunto de prueba) y se compara lo que predice cada modelo, con sus probabilidades.
- **Rendimiento de los modelos:** F1 macro, F1 por clase, matrices de confusión, tiempos y curvas de entrenamiento. Cada bloque se puede mostrar u ocultar desde el panel lateral.

La primera vez que se clasifica una foto con un modelo clásico, torchvision descarga los pesos de ImageNet y necesita conexión a internet.

La app necesita estos archivos en el proyecto: `data/splits/split.csv`, `data/processed/images_224/`, `data/embeddings/`, `resultados/` y `models/` (los `.joblib` de los modelos clásicos y, en `models/finetuning/<modelo>/`, `mejor.pt`, `metricas.json`, `historial.csv`, `real_test.npy` y `pred_test.npy`, más `clases.json` en `models/finetuning/`). Si falta algún modelo, la app lo avisa y sigue funcionando con los demás.

### 4. Compilar el informe

El informe está en LaTeX (`informe/informe.tex`). Requiere `pdflatex` con el paquete de idioma español (`texlive-lang-spanish`):

```bash
cd informe
pdflatex informe.tex
pdflatex informe.tex   # dos veces, para que el índice quede bien generado
```

Las figuras que usa son las que genera el análisis en `figuras/`.

## Datos

- **Train.csv:** 26,068 filas con `ID`, `filename`, `growth_stage` (Siembra, Vegetativo, Floración, Madurez), `damage` (8 categorías), `extent` (severidad, 0 a 100%; solo es distinta de 0 cuando el daño es sequía) y `season` (LR2020, LR2021, SR2020, SR2021).
- **Test.csv:** 8,663 filas con las mismas columnas, sin `extent` (la variable que se pedía predecir en la competencia original). Sí trae el tipo de daño, pero el 99% de sus fotos comparten campo con las de `Train.csv`, así que no sirve como conjunto de evaluación independiente. La evaluación de este proyecto usa la partición agrupada de `Train.csv`.
- **Imágenes:** `train.zip` (~7 GB) y `test.zip` (~2.2 GB) se descargan desde Google Drive con `src/descarga_imagenes.py`. Para el modelado se usan fotos redimensionadas a 224×224.

## Notas

- Los pesos de los modelos, los embeddings y las imágenes no se versionan en Git. Quien clone el repositorio debe correr los notebooks en el orden indicado para regenerarlos.
- Los `.joblib` de los modelos clásicos dependen de la versión de scikit-learn con la que se entrenaron. Si se carga con otra versión pueden aparecer avisos o errores; en ese caso conviene reentrenarlos con el notebook 05.
- Si `pip` falla con errores de certificado SSL o 403, suele ser un problema de la red; probar desde otra red resolvió el problema en este proyecto.