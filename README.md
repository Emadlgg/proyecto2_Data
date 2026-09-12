# Proyecto 2 - Análisis Exploratorio: CGIAR Eyes on the Ground

Análisis exploratorio de datos del reto [CGIAR Eyes on the Ground Challenge](https://zindi.africa/competitions/cgiar-eyes-on-the-ground-challenge) (Zindi), que busca detectar daño por sequía y otras causas en imágenes de cultivos tomadas por agricultores en Kenia, como parte de un sistema de seguro agrícola basado en imágenes (*picture-based insurance*).

**Curso:** Data Science - UVG
**Integrante:** Osman Emanuel de León García
**Entrega:** Septiembre de 2026

## Estructura del proyecto

```
├── data/                      CSVs (Train, Test, SampleSubmission) y muestra de imágenes
│   └── raw/                   Zips descargados e imágenes extraídas (no se versiona en git)
├── src/                       Funciones reutilizables
│   ├── descarga_imagenes.py     Descarga los zips desde Drive y extrae muestras puntuales
│   ├── limpieza.py               Carga de datos, mapeo de etiquetas, filtros
│   └── visualizacion.py          Histogramas, boxplots, barras, grillas de imágenes
├── notebook_eda.ipynb         Notebook principal del análisis exploratorio
├── figuras/                   Gráficos exportados en PNG (usados en notebook e informe)
├── informe/
│   ├── informe.tex              Código fuente LaTeX del informe
│   └── informe_proyecto2_cgiar.pdf   Informe final compilado
├── presentacion/
│   └── proyecto2_cgiar.pptx     Presentación final
├── requirements.txt
└── .gitignore
```

## Cómo correr el notebook

1. Crear un entorno virtual e instalar dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate        # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Colocar `Train.csv`, `Test.csv` y `SampleSubmission.csv` en `data/` (se descargan desde la pestaña "Data" del reto en Zindi).

3. Poner los `FILE_IDS` reales de Google Drive en `src/descarga_imagenes.py`.

4. Levantar Jupyter y correr las celdas en orden:
   ```bash
   jupyter notebook notebook_eda.ipynb
   ```
   La sección de descarga de imágenes trae automáticamente `train.zip` desde Drive la primera vez (puede tardar varios minutos por el tamaño del archivo) y extrae solo una muestra balanceada por clase.

## Cómo compilar el informe

El informe está en LaTeX (`informe/informe.tex`). Para compilarlo se necesita `pdflatex` con el paquete de idioma español (`texlive-lang-spanish`):

```bash
cd informe
pdflatex informe.tex
pdflatex informe.tex   # se corre dos veces para que el índice quede bien generado
```

Las figuras que usa el informe son las mismas que genera el notebook en `figuras/`, así que hay que correr el notebook al menos una vez antes de compilar.

## Datos

- **Train.csv**: 26,068 filas — `ID`, `filename`, `growth_stage` (Siembra/Vegetativo/Floración/Madurez), `damage` (8 categorías, entre ellas Sequía), `extent` (0-100%, solo distinto de 0 cuando `damage` es Sequía), `season` (LR2020/LR2021/SR2020/SR2021).
- **Test.csv**: 8,663 filas, sin la columna `extent`.
- Las imágenes completas (train.zip ~6.5GB, test.zip ~2.2GB) se descargan aparte vía Google Drive; solo se extrae una muestra por clase para el análisis visual.

## Contenido del análisis

1. Investigación previa
2. Situación problemática, problema científico y objetivos
3. Descripción de los datos
4. Limpieza y preprocesamiento
5. EDA cuantitativo — severidad del daño por sequía (`extent`), outliers, estadística por temporada y etapa
6. EDA categórico y visual — frecuencias, tablas cruzadas, Cramér's V, muestra de imágenes por clase
7. Hallazgos y conclusiones