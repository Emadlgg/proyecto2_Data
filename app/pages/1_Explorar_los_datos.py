import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency

from src.app_comun import (
    COLOR_CLASE, ORDEN_ETAPAS, ORDEN_TEMPORADAS, archivos_disponibles, exigir_datos, ruta_foto,
)

st.set_page_config(page_title="Explorar los datos", page_icon="🌽", layout="wide")

df = exigir_datos()

st.title("Explorar los datos")

# ----------------------------------------------------------------- filtros
with st.sidebar:
    st.header("Filtros")
    temporadas = st.multiselect("Temporada", ORDEN_TEMPORADAS, default=ORDEN_TEMPORADAS)
    etapas = st.multiselect("Etapa de crecimiento", ORDEN_ETAPAS, default=ORDEN_ETAPAS)
    todos_los_danos = sorted(df["daño"].unique())
    danos = st.multiselect("Tipo de daño", todos_los_danos, default=todos_los_danos)

filtrado = df[df["temporada"].isin(temporadas) & df["etapa"].isin(etapas) & df["daño"].isin(danos)]
if filtrado.empty:
    st.warning("Con estos filtros no queda ninguna foto. Probá con más opciones seleccionadas.")
    st.stop()

st.caption(f"{len(filtrado):,} de {len(df):,} fotos con los filtros actuales")

tab_dist, tab_sev, tab_rel, tab_fotos = st.tabs(
    ["Distribución", "Severidad de la sequía", "Relaciones entre variables", "Fotos de ejemplo"]
)

# ------------------------------------------------------------ distribución
with tab_dist:
    izq, der = st.columns(2)

    por_dano = filtrado.groupby(["daño", "clase4"]).size().reset_index(name="fotos").sort_values("fotos")
    fig = px.bar(
        por_dano, x="fotos", y="daño", color="clase4", color_discrete_map=COLOR_CLASE,
        orientation="h", text="fotos", labels={"daño": "", "fotos": "Fotos", "clase4": "Grupo"},
        title="Fotos por tipo de daño",
    )
    fig.update_layout(height=380, legend_title_text="Grupo de 4 clases")
    izq.plotly_chart(fig)

    por_etapa = filtrado["etapa"].value_counts().reindex(ORDEN_ETAPAS).dropna().reset_index()
    por_etapa.columns = ["etapa", "fotos"]
    fig = px.bar(
        por_etapa, x="etapa", y="fotos", text="fotos", color_discrete_sequence=["#2E3A46"],
        labels={"etapa": "", "fotos": "Fotos"}, title="Fotos por etapa de crecimiento",
    )
    fig.update_layout(height=380)
    der.plotly_chart(fig)

    st.caption(
        "Los colores indican el grupo de 4 clases al que pertenece cada tipo de daño. "
        "Bueno y Maleza dominan el conjunto; las clases de \"Otro daño\" tienen muy pocas fotos."
    )

# ------------------------------------------------------------- severidad
with tab_sev:
    sequia = filtrado[filtrado["damage"] == "DR"]
    if sequia.empty:
        st.info("Con los filtros actuales no hay fotos de sequía. Incluí \"Sequía\" en el filtro de tipo de daño.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Fotos de sequía", f"{len(sequia):,}")
        m2.metric("Severidad media", f"{sequia['extent'].mean():.1f}%")
        m3.metric("Severidad mediana", f"{sequia['extent'].median():.0f}%")

        comparar = st.radio("Comparar la severidad por", ["Temporada", "Etapa de crecimiento", "Nada"], horizontal=True)

        izq, der = st.columns(2)
        fig = px.histogram(
            sequia, x="extent", nbins=10, color_discrete_sequence=[COLOR_CLASE["Sequía"]],
            labels={"extent": "Severidad (%)", "count": "Fotos"}, title="Distribución de la severidad",
        )
        fig.update_layout(height=380, bargap=0.1, yaxis_title="Fotos")
        izq.plotly_chart(fig)

        if comparar == "Nada":
            fig = px.box(sequia, y="extent", color_discrete_sequence=[COLOR_CLASE["Sequía"]],
                         labels={"extent": "Severidad (%)"}, title="Severidad de la sequía")
            columna_grupo = None
        else:
            columna_grupo, orden = ("temporada", ORDEN_TEMPORADAS) if comparar == "Temporada" else ("etapa", ORDEN_ETAPAS)
            fig = px.box(
                sequia, x=columna_grupo, y="extent", category_orders={columna_grupo: orden},
                color_discrete_sequence=[COLOR_CLASE["Sequía"]],
                labels={"extent": "Severidad (%)", columna_grupo: comparar}, title=f"Severidad por {comparar.lower()}",
            )
        fig.update_layout(height=380)
        der.plotly_chart(fig)

        if columna_grupo:
            conteos = sequia[columna_grupo].value_counts()
            chicos = conteos[conteos < 50]
            if not chicos.empty:
                st.caption(
                    "Cuidado: pocas fotos en " + ", ".join(f"{g} ({n})" for g, n in chicos.items())
                    + ". Con tan pocos datos la comparación no es confiable."
                )

# ------------------------------------------------------------- relaciones
with tab_rel:
    cruzar = st.radio("Cruzar el tipo de daño con", ["Temporada", "Etapa de crecimiento"], horizontal=True)
    col, orden = ("temporada", ORDEN_TEMPORADAS) if cruzar == "Temporada" else ("etapa", ORDEN_ETAPAS)

    tabla = pd.crosstab(filtrado["daño"], filtrado[col], normalize="columns").mul(100)
    tabla = tabla[[c for c in orden if c in tabla.columns]]
    fig = px.imshow(
        tabla, text_auto=".1f", aspect="auto", color_continuous_scale="Cividis",
        labels={"x": cruzar, "y": "", "color": "% de la columna"},
        title=f"Composición del daño dentro de cada {cruzar.lower()} (%)",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig)

    st.subheader("Qué tan asociadas están las variables")

    def cramers_v(a, b):
        t = pd.crosstab(a, b)
        chi2 = chi2_contingency(t)[0]
        return float(np.sqrt((chi2 / t.values.sum()) / (min(t.shape) - 1)))

    c1, c2 = st.columns(2)
    c1.metric("Daño × temporada (V de Cramér)", f"{cramers_v(df['daño'], df['temporada']):.2f}",
              help="Calculada con todas las fotos, sin filtros. 0 = sin relación, 1 = relación perfecta.")
    c2.metric("Daño × etapa (V de Cramér)", f"{cramers_v(df['daño'], df['etapa']):.2f}",
              help="Calculada con todas las fotos, sin filtros. 0 = sin relación, 1 = relación perfecta.")
    st.caption("La etapa de crecimiento se asocia bastante más con el tipo de daño que la temporada.")

# ----------------------------------------------------------------- fotos
with tab_fotos:
    izq, der, _ = st.columns([1, 1, 2])
    clase = izq.selectbox("Tipo de daño", sorted(filtrado["daño"].unique()))
    cuantas = der.select_slider("Cuántas fotos", options=[4, 8, 12], value=8)

    if "semilla_fotos" not in st.session_state:
        st.session_state["semilla_fotos"] = 0
    if st.button("Mostrar otras fotos"):
        st.session_state["semilla_fotos"] += 1

    candidatas = filtrado[filtrado["daño"] == clase]
    candidatas = candidatas[candidatas["archivo_224"].isin(archivos_disponibles())]

    if candidatas.empty:
        st.info("No hay fotos de este tipo en disco para los filtros actuales (las fotos redimensionadas salen del notebook 03).")
    else:
        rng = np.random.default_rng(st.session_state["semilla_fotos"])
        elegidas = candidatas.iloc[rng.choice(len(candidatas), size=min(cuantas, len(candidatas)), replace=False)]
        columnas = st.columns(4)
        for i, (_, fila) in enumerate(elegidas.iterrows()):
            texto = f"{fila['etapa']} · {fila['temporada']}"
            if fila["damage"] == "DR":
                texto += f" · severidad {fila['extent']}%"
            columnas[i % 4].image(str(ruta_foto(fila["archivo_224"])), caption=texto)
