import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.app_comun import (
    CLASES, COLOR_AZAR, COLOR_TIPO, RUTA_EMBEDDINGS, RUTA_MODELOS, cargar_historial, cargar_predicciones_finetuning,
    cargar_resultados,
)
from src.app_modelos import predicciones_clasico_test

st.set_page_config(page_title="Rendimiento de los modelos", page_icon="🌽", layout="wide")

st.title("Rendimiento de los modelos")
st.markdown(
    "Todas las métricas son sobre el **conjunto de prueba**, que los modelos no vieron al entrenar ni al elegir "
    "sus parámetros. La métrica principal es el **F1 macro**, que promedia el F1 de cada clase sin que dominen "
    "las clases grandes."
)

experimento = st.radio("Experimento", ["4 clases (principal)", "8 clases (secundario)"], horizontal=True)
objetivo = "clase4" if experimento.startswith("4") else "damage"

tabla, referencias = cargar_resultados(objetivo)
if tabla is None:
    st.error(f"No se encontró `resultados/clasicos_{objetivo}.csv`. Corré el notebook 05 (modelos clásicos).")
    st.stop()

if objetivo == "damage":
    st.info(
        "Con 8 clases, Inundación y Viento tienen solo 2 y 5 fotos de prueba, y Enfermedad 18. "
        "Sus métricas son muy ruidosas, por eso este es el experimento secundario. "
        "Los modelos fine-tuneados se entrenaron solo con 4 clases."
    )

columnas_f1 = [c for c in tabla.columns if c.startswith("f1_") and c not in ("f1_macro_val", "f1_macro_test")]
columnas_f1 = [c for c in columnas_f1 if tabla[c].notna().any()]
mejor = tabla.loc[tabla["f1_macro_test"].idxmax()]
azar = referencias.get("baseline_stratified")

# ------------------------------------------------------------- qué mostrar
with st.sidebar:
    st.header("Qué mostrar")
    ver_f1 = st.toggle("F1 macro por modelo", value=True)
    ver_clase = st.toggle("F1 por clase", value=True)
    ver_conf = st.toggle("Matriz de confusión", value=True)
    ver_tiempos = st.toggle("Tiempos", value=True)
    ver_curvas = st.toggle("Curvas de entrenamiento", value=True, disabled=(objetivo != "clase4"))
    ver_tabla = st.toggle("Tabla completa", value=False)

# ---------------------------------------------------------------- F1 macro
if ver_f1:
    st.subheader("F1 macro por modelo")
    datos = tabla.sort_values("f1_macro_test")
    fig = px.bar(
        datos, x="f1_macro_test", y="etiqueta", color="tipo", color_discrete_map=COLOR_TIPO, orientation="h",
        text=datos["f1_macro_test"].map(lambda v: f"{v:.3f}"),
        labels={"f1_macro_test": "F1 macro (test)", "etiqueta": "", "tipo": "Tipo de modelo"},
    )
    if azar is not None:
        fig.add_vline(x=azar["f1_macro_test"], line_dash="dash", line_color=COLOR_AZAR,
                      annotation_text="al azar", annotation_position="top")
    fig.update_layout(height=140 + 42 * len(datos), xaxis_range=[0, max(0.7, datos["f1_macro_test"].max() * 1.1)])
    st.plotly_chart(fig)
    texto = f"El mejor modelo es **{mejor['etiqueta']}**, con F1 macro de {mejor['f1_macro_test']:.3f}"
    if azar is not None:
        texto += f". Un clasificador que respondiera al azar según la proporción de clases daría {azar['f1_macro_test']:.3f}"
    st.markdown(texto + ".")

# --------------------------------------------------------------- por clase
if ver_clase:
    st.subheader("F1 por clase")
    orden = tabla.sort_values("f1_macro_test", ascending=False)
    matriz = orden.set_index("etiqueta")[columnas_f1]
    matriz.columns = [c.replace("f1_", "") for c in columnas_f1]
    fig = px.imshow(
        matriz, text_auto=".2f", aspect="auto", color_continuous_scale="Cividis", zmin=0, zmax=1,
        labels={"x": "", "y": "", "color": "F1"},
    )
    fig.update_layout(height=140 + 45 * len(matriz))
    st.plotly_chart(fig)
    st.caption("Las clases con pocas fotos (como \"Otro daño\") son las más difíciles para todos los modelos.")

# ---------------------------------------------------------------- confusión
if ver_conf:
    st.subheader("Matriz de confusión")

    def predicciones(fila):
        if fila["tipo"] == "Fine-tuning":
            return cargar_predicciones_finetuning(fila["modelo"])
        archivo = RUTA_MODELOS / f"{fila['embedding']}_{fila['modelo']}_{objetivo}.joblib"
        if (RUTA_EMBEDDINGS / "indice.csv").exists() and archivo.exists():
            return predicciones_clasico_test(fila["embedding"], fila["modelo"], objetivo)
        return None

    etiquetas = tabla.sort_values("f1_macro_test", ascending=False)["etiqueta"].tolist()
    izq, der, _ = st.columns([2, 2, 2])
    elegido = izq.selectbox("Modelo", etiquetas, index=0)
    modo = der.radio("Mostrar", ["Porcentaje por fila", "Cantidad de fotos"], horizontal=True)

    resultado = predicciones(tabla[tabla["etiqueta"] == elegido].iloc[0])
    if resultado is None:
        st.info("No se encontraron las predicciones de este modelo (hacen falta sus archivos en `models/` y `data/embeddings/`).")
    else:
        real, pred = resultado
        clases = CLASES if objetivo == "clase4" else sorted(set(real))
        porcentaje = modo.startswith("Porcentaje")
        cm = confusion_matrix(real, pred, labels=clases, normalize="true" if porcentaje else None)
        fig = px.imshow(
            cm, x=clases, y=clases, text_auto=".0%" if porcentaje else "d", color_continuous_scale="Cividis",
            labels={"x": "Predicción", "y": "Clase real", "color": ""}, zmin=0, zmax=1 if porcentaje else None,
        )
        fig.update_layout(height=420 if objetivo == "clase4" else 620)
        st.plotly_chart(fig)
        st.caption("Cada fila suma 100%: muestra, de las fotos de una clase real, cuántas se predijeron como cada clase."
                   if porcentaje else "Cantidad de fotos de prueba por clase real y clase predicha.")

# ------------------------------------------------------------------ tiempos
if ver_tiempos:
    st.subheader("Tiempos")
    datos = tabla.dropna(subset=["tiempo_entrenamiento_s"]).sort_values("tiempo_entrenamiento_s")
    log = st.checkbox("Escala logarítmica", value=True)
    izq, der = st.columns(2)
    for columna, titulo, contenedor in [
        ("tiempo_entrenamiento_s", "Entrenamiento (segundos)", izq),
        ("tiempo_prediccion_s", "Predicción sobre test (segundos)", der),
    ]:
        fig = px.bar(
            datos.sort_values(columna), x=columna, y="etiqueta", color="tipo", color_discrete_map=COLOR_TIPO,
            orientation="h", log_x=log, labels={columna: titulo, "etiqueta": "", "tipo": ""}, title=titulo,
        )
        fig.update_layout(height=140 + 42 * len(datos), showlegend=False)
        contenedor.plotly_chart(fig)
    st.caption(
        "Para comparar con cuidado: los tiempos de los modelos clásicos no incluyen sacar los embeddings con la CNN "
        "(se hace una sola vez, aparte). Los del fine-tuning incluyen leer las fotos y pasarlas por la red completa."
    )

# ------------------------------------------------------------------- curvas
if ver_curvas and objetivo == "clase4":
    historiales = {n: cargar_historial(n) for n in ("resnet50", "efficientnet_b0")}
    historiales = {n: h for n, h in historiales.items() if h is not None}
    if historiales:
        st.subheader("Curvas de entrenamiento (fine-tuning)")
        nombres = {"resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0"}
        largo = pd.concat([h.assign(modelo=nombres[n]) for n, h in historiales.items()], ignore_index=True)
        fase2 = largo.loc[largo["fase"] == 2, "epoca"].min()

        izq, der = st.columns(2)
        fig = px.line(
            largo, x="epoca", y="f1_macro_val", line_dash="modelo", markers=True,
            color_discrete_sequence=["#2E3A46"], title="F1 macro en validación",
            labels={"epoca": "Época", "f1_macro_val": "F1 macro (validación)", "modelo": "Modelo"},
        )
        fig.add_vline(x=fase2 - 0.5, line_dash="dot", line_color=COLOR_AZAR, annotation_text="empieza la fase 2")
        izq.plotly_chart(fig)

        perdidas = largo.melt(
            id_vars=["epoca", "modelo"], value_vars=["perdida_train", "perdida_val"],
            var_name="conjunto", value_name="perdida",
        )
        perdidas["conjunto"] = perdidas["conjunto"].map({"perdida_train": "entrenamiento", "perdida_val": "validación"})
        perdidas["serie"] = perdidas["modelo"] + " · " + perdidas["conjunto"]
        fig = px.line(
            perdidas, x="epoca", y="perdida", color="conjunto", line_dash="modelo", markers=True,
            color_discrete_map={"entrenamiento": "#2E3A46", "validación": "#E69F00"}, title="Pérdida",
            labels={"epoca": "Época", "perdida": "Pérdida", "conjunto": "Conjunto", "modelo": "Modelo"},
        )
        fig.add_vline(x=fase2 - 0.5, line_dash="dot", line_color=COLOR_AZAR)
        der.plotly_chart(fig)
        st.caption(
            "Primero (fase 1) solo aprende la capa final; después (fase 2) aprende toda la red. Cuando la pérdida de "
            "entrenamiento sigue bajando y la de validación sube, el modelo empieza a memorizar las fotos de entrenamiento."
        )

# ------------------------------------------------------------ tabla completa
if ver_tabla:
    st.subheader("Tabla completa")
    columnas = ["etiqueta", "tipo", "f1_macro_val", "f1_macro_test", "balanced_accuracy_test", "accuracy_test",
                *columnas_f1, "tiempo_entrenamiento_s", "tiempo_prediccion_s"]
    st.dataframe(tabla.sort_values("f1_macro_test", ascending=False)[columnas].round(3), hide_index=True)
