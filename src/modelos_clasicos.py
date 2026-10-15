"""
Modelos clasicos (regresion logistica, SVM y Random Forest) para clasificar
a partir de embeddings de CNN preentrenadas.

El desbalance de clases se maneja con class_weight="balanced" en los tres.
"""

import time

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Parametros que se prueban en validacion para cada modelo
GRILLAS = {
    "regresion_logistica": [{"C": 0.0003}, {"C": 0.001}, {"C": 0.01}, {"C": 0.1}],
    "svm": [{"C": 1.0}, {"C": 10.0}, {"C": 30.0}],
    "random_forest": [{"min_samples_leaf": 1}, {"min_samples_leaf": 3}],
}

PCA_SVM = 256  # el kernel RBF escala mal con 2048 dimensiones; se reduce con PCA


def crear_pipeline(modelo, semilla=42, **params):
    """Devuelve un Pipeline de sklearn listo para entrenar."""
    if modelo == "regresion_logistica":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=semilla, **params)
        pasos = [("escala", StandardScaler()), ("clf", clf)]
    elif modelo == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=semilla, **params)
        pasos = [
            ("escala", StandardScaler()),
            ("pca", PCA(n_components=PCA_SVM, random_state=semilla)),
            ("clf", clf),
        ]
    elif modelo == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=semilla, **params
        )
        pasos = [("clf", clf)]  # los arboles no necesitan escalar
    else:
        raise ValueError(f"Modelo desconocido: {modelo}")
    return Pipeline(pasos)


def crear_baseline(estrategia, semilla=42):
    """Clasificador de referencia: 'most_frequent' o 'stratified'."""
    return DummyClassifier(strategy=estrategia, random_state=semilla)


def evaluar(modelo, X, y):
    """Predice sobre X y devuelve las metricas junto con el tiempo de prediccion."""
    t0 = time.time()
    pred = modelo.predict(X)
    t_pred = time.time() - t0

    clases = sorted(set(y))
    f1_clase = f1_score(y, pred, labels=clases, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_por_clase": dict(zip(clases, f1_clase)),
        "tiempo_prediccion_s": t_pred,
        "pred": pred,
    }
