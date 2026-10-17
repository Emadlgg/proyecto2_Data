import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src.app_comun import (
    CLASES, COLOR_CLASE, ORDEN_ETAPAS, archivos_disponibles, cargar_resultados, chip, exigir_datos, ruta_foto,
)
from src.app_modelos import DependenciaFaltante, ErrorDeFoto, clasificar, opciones_modelos, preparar_imagen

st.set_page_config(page_title="Clasificar una foto", page_icon="🌽", layout="wide")

df = exigir_datos()
opciones = opciones_modelos()
disponibles = {k: v for k, v in opciones.items() if v["disponible"]}

st.title("Clasificar una foto")
st.markdown(
    "Subí una foto de un cultivo, o elegí una del conjunto de prueba, y mirá qué dice cada modelo. "
    "La foto se ajusta sola al formato que usan los modelos."
)

if not disponibles:
    st.error("No se encontraron modelos entrenados en la carpeta `models/`. Corré los notebooks 05 y 06 primero.")
    st.stop()

no_encontrados = [v["nombre"] for v in opciones.values() if not v["disponible"]]
if no_encontrados:
    st.caption("No se encontraron en disco: " + ", ".join(no_encontrados))


def mostrar_resultado(res, etapa):
    filas = res["filas"]
    real = res["real"]
    validas = [f for f in filas if f["error"] is None]

    if real:
        st.markdown(f"**Clase real de la foto:** {chip(real, COLOR_CLASE[real])}", unsafe_allow_html=True)

    if validas:
        votos = Counter(f["clase"] for f in validas)
        n_top = max(votos.values())
        lideres = [c for c, n in votos.items() if n == n_top]
        chips = " ".join(chip(c, COLOR_CLASE[c]) for c in lideres)
        if len(lideres) == 1:
            st.markdown(f"**{n_top} de {len(validas)} modelos coinciden en:** {chips}", unsafe_allow_html=True)
        else:
            st.markdown(f"**Empate entre:** {chips} ({n_top} modelos cada una)", unsafe_allow_html=True)

        encabezado = "<th style='text-align:left'>Modelo</th><th style='text-align:left'>Predicción</th><th style='text-align:right'>Confianza</th>"
        if real:
            encabezado += "<th style='text-align:center'>¿Acierta?</th>"
        cuerpo = ""
        for f in validas:
            confianza = f"{max(f['probas'].values()) * 100:.0f}%" if f["probas"] else "—"
            fila_html = (
                f"<td>{opciones[f['id']]['nombre']}</td>"
                f"<td>{chip(f['clase'], COLOR_CLASE[f['clase']])}</td>"
                f"<td style='text-align:right'>{confianza}</td>"
            )
            if real:
                fila_html += f"<td style='text-align:center'>{'✓' if f['clase'] == real else '✗'}</td>"
            cuerpo += f"<tr style='border-bottom:1px solid #E3DED4'>{fila_html}</tr>"
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse'><thead><tr style='border-bottom:2px solid #2E3A46'>"
            f"{encabezado}</tr></thead><tbody>{cuerpo}</tbody></table>",
            unsafe_allow_html=True,
        )

        st.write("")
        if st.toggle("Mostrar las probabilidades de cada modelo", value=True):
            largo = pd.DataFrame(
                [
                    {"modelo": opciones[f["id"]]["nombre"], "clase": c, "probabilidad": p}
                    for f in validas if f["probas"] for c, p in f["probas"].items()
                ]
            )
            if not largo.empty:
                fig = px.bar(
                    largo, x="probabilidad", y="modelo", color="clase", orientation="h",
                    color_discrete_map=COLOR_CLASE, category_orders={"clase": CLASES},
                    labels={"probabilidad": "Probabilidad", "modelo": "", "clase": "Clase"},
                )
                fig.update_layout(barmode="stack", xaxis_tickformat=".0%", height=120 + 45 * largo["modelo"].nunique(),
                                  margin=dict(t=10))
                st.plotly_chart(fig)
            sin_probas = [opciones[f["id"]]["nombre"] for f in validas if not f["probas"]]
            if sin_probas:
                st.caption(f"{', '.join(sin_probas)} no entrega probabilidades, solo la clase predicha.")

    for f in filas:
        if f["error"]:
            st.warning(f"{opciones[f['id']]['nombre']}: {f['error']}")

    if etapa != "No sé":
        pct_etapa = (df[df["etapa"] == etapa]["damage"] == "DR").mean() * 100
        pct_total = (df["damage"] == "DR").mean() * 100
        st.info(
            f"En el conjunto de datos, {pct_etapa:.0f}% de las fotos de la etapa **{etapa}** muestran sequía "
            f"(contra {pct_total:.0f}% en general). Es un dato de contexto, los modelos no lo usan."
        )

    tabla, _ = cargar_resultados("clase4")
    if tabla is not None:
        st.caption(
            f"Ningún modelo es perfecto: en el conjunto de prueba, el mejor acierta alrededor de "
            f"{tabla['accuracy_test'].max() * 100:.0f}% de las fotos. Tomá la predicción como una orientación."
        )


col_in, col_out = st.columns([1, 1.4], gap="large")

with col_in:
    modo = st.radio("Origen de la foto", ["Subir una foto", "Elegir una foto de prueba"], horizontal=True)

    imagen, real, etapa_real, firma = None, None, None, None

    if modo == "Subir una foto":
        archivo = st.file_uploader("Foto del cultivo", type=["jpg", "jpeg", "png", "webp"])
        if archivo is not None:
            try:
                imagen = preparar_imagen(archivo)
                firma = f"subida:{archivo.name}:{archivo.size}"
            except ErrorDeFoto as e:
                st.error(str(e))
    else:
        pool = df[(df["split"] == "test") & df["archivo_224"].isin(archivos_disponibles())]
        if pool.empty:
            st.info("No hay fotos de prueba en disco (las fotos redimensionadas salen del notebook 03).")
        else:
            filtro = st.selectbox("Clase real de la foto", ["Cualquiera"] + CLASES)
            if "semilla_test" not in st.session_state:
                st.session_state["semilla_test"] = 0
            if st.button("Otra foto al azar"):
                st.session_state["semilla_test"] += 1
            candidatas = pool if filtro == "Cualquiera" else pool[pool["clase4"] == filtro]
            fila = candidatas.iloc[np.random.default_rng(st.session_state["semilla_test"]).integers(len(candidatas))]
            imagen = Image.open(ruta_foto(fila["archivo_224"])).convert("RGB")
            real, etapa_real = fila["clase4"], fila["etapa"]
            firma = f"test:{fila['archivo_224']}"

    if imagen is not None:
        st.image(imagen, caption="Así ve la foto el modelo (224×224)", width=280)

    opciones_etapa = ["No sé"] + ORDEN_ETAPAS
    etapa = st.selectbox(
        "Etapa de crecimiento (opcional)", opciones_etapa,
        index=opciones_etapa.index(etapa_real) if etapa_real else 0, key=f"etapa_{firma}",
        help="Solo sirve para mostrar contexto del conjunto de datos. Los modelos clasifican únicamente con la imagen.",
    )

    nombres = {k: v["nombre"] for k, v in disponibles.items()}
    todos = st.checkbox("Usar todos los modelos")
    elegidos = st.multiselect(
        "Modelos", list(nombres), default=[k for k in nombres if k.startswith("ft:")],
        format_func=nombres.get, disabled=todos,
    )
    if todos:
        elegidos = list(nombres)

    apretado = st.button("Clasificar", type="primary", disabled=(imagen is None or not elegidos))

if apretado:
    with st.spinner("Clasificando..."):
        try:
            st.session_state["resultado"] = {"filas": clasificar(imagen, elegidos), "firma": firma, "real": real}
        except DependenciaFaltante as e:
            st.session_state.pop("resultado", None)
            col_out.error(str(e))

with col_out:
    res = st.session_state.get("resultado")
    if res is None or res["firma"] != firma:
        st.info("Elegí una foto y uno o más modelos, y apretá **Clasificar**.")
    else:
        mostrar_resultado(res, etapa)
