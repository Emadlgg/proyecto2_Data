import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st

from src.app_comun import COLOR_CLASE, ORDEN_ETAPAS, cargar_resultados, exigir_datos

st.set_page_config(page_title="Daño por sequía en cultivos", page_icon="🌽", layout="wide")

df = exigir_datos()

st.title("Detección de daño por sequía en fotos de cultivos")
st.caption("Proyecto 2 · Data Science · UVG · Reto CGIAR Eyes on the Ground")

st.markdown(
    """
    En Kenia, muchos pequeños agricultores aseguran sus cosechas y reclaman con fotos tomadas con el celular.
    Revisar cada foto a mano es lento y retrasa los pagos justo cuando más se necesitan.
    Esta aplicación muestra qué se puede aprender de esas fotos y qué tan bien un modelo puede
    reconocer el tipo de daño, en especial la **sequía**.
    """
)

# ------------------------------------------------------------ cifras clave
sequia = df[df["damage"] == "DR"]
pct_sequia = len(sequia) / len(df) * 100
por_etapa = sequia["etapa"].value_counts(normalize=True)
etapa_top = por_etapa.idxmax()
pct_etapa_top = por_etapa.max() * 100

tabla, _ = cargar_resultados("clase4")
mejor = tabla.loc[tabla["f1_macro_test"].idxmax()] if tabla is not None else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fotos analizadas", f"{len(df):,}")
c2.metric("Fotos con sequía", f"{pct_sequia:.0f}%")
c3.metric(f"De la sequía, en {etapa_top}", f"{pct_etapa_top:.0f}%")
if mejor is not None:
    c4.metric("Mejor modelo (F1 macro)", f"{mejor['f1_macro_test']:.2f}", help=mejor["etiqueta"])

# --------------------------------------------------------------- hallazgo
st.subheader("El hallazgo principal: la sequía se concentra en una etapa del cultivo")

pct_por_etapa = (
    df.assign(es_sequia=df["damage"] == "DR").groupby("etapa")["es_sequia"].mean().mul(100)
    .reindex(ORDEN_ETAPAS).reset_index(name="pct")
)
fig = px.bar(
    pct_por_etapa, x="etapa", y="pct", text=pct_por_etapa["pct"].map(lambda v: f"{v:.0f}%"),
    color_discrete_sequence=[COLOR_CLASE["Sequía"]],
    labels={"etapa": "Etapa de crecimiento", "pct": "% de fotos con sequía"},
)
fig.update_traces(textposition="outside")
fig.update_layout(yaxis_range=[0, max(60, pct_por_etapa["pct"].max() * 1.2)], margin=dict(t=20), height=360)
st.plotly_chart(fig)

fila_top = pct_por_etapa.loc[pct_por_etapa["pct"].idxmax()]
st.markdown(
    f"En **{fila_top['etapa']}**, {fila_top['pct']:.0f}% de las fotos muestran sequía, "
    f"y esa etapa reúne el {pct_etapa_top:.0f}% de todos los casos de sequía del conjunto de datos."
)

# ------------------------------------------------------------ navegación
st.subheader("Qué hay en cada página")
st.markdown(
    """
    - **Explorar los datos:** filtrá por temporada, etapa y tipo de daño, y mirá cómo cambian la distribución,
      la severidad de la sequía y las fotos de ejemplo.
    - **Clasificar una foto:** subí una foto (o elegí una del conjunto de prueba) y compará lo que dice cada modelo.
    - **Rendimiento de los modelos:** qué tan bien funciona cada modelo y cuánto tarda, con gráficos que podés mostrar u ocultar.
    """
)

with st.expander("Cómo se plantearon las clases"):
    st.markdown(
        """
        El conjunto original distingue 8 tipos de daño, pero cinco de ellos (enfermedad, inundación, deficiencia de
        nutrientes, plaga y viento) suman menos del 3% de las fotos y tienen muy pocos ejemplos para entrenar o evaluar.
        Por eso el experimento principal usa 4 clases: **Bueno, Maleza, Sequía y Otro daño**. Los resultados con las
        8 clases originales se muestran como experimento secundario en la página de rendimiento.
        """
    )
