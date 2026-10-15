"""
Fine-tuning de CNN preentrenadas (EfficientNet-B0 y ResNet50) para clasificar
el dano en las fotos de cultivos.

Estrategia en dos fases:
  1. Solo se entrena la capa final (la base queda congelada).
  2. Se entrena toda la red con una tasa de aprendizaje mas baja, con parada
     temprana segun el F1 macro en validacion.

Se guarda un checkpoint despues de cada epoca, asi que si la sesion de Colab
se corta, al volver a correr se continua desde la ultima epoca terminada.
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.embeddings import DESV_IMAGENET, MEDIA_IMAGENET

# Nombre de la capa final de cada modelo
CABEZA = {"resnet50": "fc", "efficientnet_b0": "classifier"}


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


# ---------------------------------------------------------------- datos

def transformaciones(entrenamiento):
    """
    Con aumento de datos suave para train, sin aumento para val y test.

    No se toca el tono ni la saturacion, porque el color es justo la senal de
    la sequia. Tampoco se voltea verticalmente: el horizonte tiene sentido.
    """
    normalizar = [transforms.ToTensor(), transforms.Normalize(MEDIA_IMAGENET, DESV_IMAGENET)]
    if not entrenamiento:
        return transforms.Compose(normalizar)
    aumento = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    return transforms.Compose(aumento + normalizar)


class ImagenesEtiquetadas(Dataset):
    """Fotos de 224x224 con su clase (como numero entero)."""

    def __init__(self, carpeta, archivos, etiquetas, transform):
        self.carpeta = Path(carpeta)
        self.archivos = list(archivos)
        self.etiquetas = np.asarray(etiquetas)
        self.transform = transform

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, i):
        im = Image.open(self.carpeta / self.archivos[i]).convert("RGB")
        return self.transform(im), int(self.etiquetas[i])


def crear_loaders(carpeta, meta, columna_clase, clases, batch=64, workers=2):
    """Devuelve {'train': loader, 'val': loader, 'test': loader}."""
    indice_clase = {c: i for i, c in enumerate(clases)}
    loaders = {}
    for split in ("train", "val", "test"):
        sub = meta[meta["split"] == split]
        y = sub[columna_clase].map(indice_clase).values
        ds = ImagenesEtiquetadas(carpeta, sub["archivo_224"].tolist(), y, transformaciones(split == "train"))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),  # un lote de 1 foto rompe BatchNorm en train
            persistent_workers=workers > 0,
        )
    return loaders


def pesos_de_clase(y_train, n_clases, potencia=1.0):
    """
    Peso de cada clase para la funcion de perdida.
    potencia=1 equivale a class_weight="balanced"; potencia=0.5 es mas suave.
    """
    conteos = np.bincount(np.asarray(y_train), minlength=n_clases).astype(float)
    conteos = np.maximum(conteos, 1.0)  # evita dividir entre 0 si una clase falta (pruebas con pocas fotos)
    pesos = (conteos.sum() / (n_clases * conteos)) ** potencia
    return torch.tensor(pesos, dtype=torch.float32)


# --------------------------------------------------------------- modelo

def crear_modelo(nombre, n_clases, pesos_imagenet=True):
    """CNN preentrenada en ImageNet con la capa final reemplazada."""
    if nombre == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pesos_imagenet else None
        modelo = models.resnet50(weights=w)
        modelo.fc = nn.Linear(modelo.fc.in_features, n_clases)
    elif nombre == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pesos_imagenet else None
        modelo = models.efficientnet_b0(weights=w)
        modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, n_clases)  # conserva el Dropout
    else:
        raise ValueError(f"Modelo desconocido: {nombre}. Opciones: {tuple(CABEZA)}")
    return modelo


def configurar_fase(modelo, nombre, fase):
    """Fase 1: solo la capa final aprende. Fase 2: toda la red."""
    cabeza = CABEZA[nombre] + "."
    for n, p in modelo.named_parameters():
        p.requires_grad = (fase == 2) or n.startswith(cabeza)


def poner_modo_entrenamiento(modelo, nombre, fase):
    """En la fase 1 la base se deja en modo eval para que BatchNorm no cambie sus estadisticas."""
    modelo.train()
    if fase == 1:
        for n, m in modelo.named_children():
            if n != CABEZA[nombre]:
                m.eval()


def crear_escalador(dispositivo):
    habilitado = dispositivo.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=habilitado)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=habilitado)


# ---------------------------------------------------------- entrenamiento

def _autocast(dispositivo):
    return torch.autocast(
        device_type=dispositivo.type, dtype=torch.float16, enabled=(dispositivo.type == "cuda")
    )


def entrenar_epoca(modelo, loader, criterio, optimizador, escalador, dispositivo):
    perdida_total, total = 0.0, 0
    for x, y in loader:
        x = x.to(dispositivo, non_blocking=True)
        y = y.to(dispositivo, non_blocking=True)
        optimizador.zero_grad(set_to_none=True)
        with _autocast(dispositivo):
            salida = modelo(x)
            perdida = criterio(salida, y)
        escalador.scale(perdida).backward()
        escalador.step(optimizador)
        escalador.update()
        perdida_total += perdida.item() * len(y)
        total += len(y)
    return perdida_total / total


@torch.no_grad()
def predecir(modelo, loader, dispositivo, criterio=None):
    """Devuelve (clases reales, clases predichas, perdida media si se pasa criterio)."""
    modelo.eval()
    reales, preds = [], []
    perdida_total, total = 0.0, 0
    for x, y in loader:
        x = x.to(dispositivo, non_blocking=True)
        with _autocast(dispositivo):
            salida = modelo(x)
        if criterio is not None:
            perdida_total += criterio(salida.float(), y.to(dispositivo)).item() * len(y)
        preds.append(salida.argmax(dim=1).cpu().numpy())
        reales.append(y.numpy())
        total += len(y)
    perdida = perdida_total / total if criterio is not None else None
    return np.concatenate(reales), np.concatenate(preds), perdida


def entrenar(
    nombre,
    loaders,
    clases,
    pesos_clase,
    dispositivo,
    carpeta,
    epocas_cabeza=3,
    epocas_red=12,
    lr_cabeza=1e-3,
    lr_red=1e-4,
    paciencia=4,
    semilla=42,
):
    """
    Entrena el modelo en dos fases y devuelve (modelo con los mejores pesos,
    historial como DataFrame).

    En `carpeta` quedan:
      - ultimo.pt: checkpoint de la ultima epoca terminada (para reanudar)
      - mejor.pt: pesos de la epoca con mejor F1 macro en validacion
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_ultimo, ruta_mejor = carpeta / "ultimo.pt", carpeta / "mejor.pt"

    fijar_semilla(semilla)
    modelo = crear_modelo(nombre, len(clases)).to(dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos_clase.to(dispositivo))
    escalador = crear_escalador(dispositivo)

    historial, mejor_f1, mejor_epoca, sin_mejora, inicio, fase_previa = [], -1.0, 0, 0, 0, None
    if ruta_ultimo.exists():
        ck = torch.load(ruta_ultimo, map_location=dispositivo)
        modelo.load_state_dict(ck["modelo"])
        historial = ck["historial"]
        mejor_f1, mejor_epoca, sin_mejora = ck["mejor_f1"], ck["mejor_epoca"], ck["sin_mejora"]
        inicio, fase_previa = ck["epoca"], ck["fase"]
        print(f"Reanudando desde la epoca {inicio + 1} (mejor F1 macro val hasta ahora: {mejor_f1:.3f})")

    total_epocas = epocas_cabeza + epocas_red
    fase_actual, optimizador, scheduler = None, None, None

    for ep in range(inicio, total_epocas):
        fase = 1 if ep < epocas_cabeza else 2
        if fase == 2 and sin_mejora >= paciencia:
            print("Parada temprana: la validacion dejo de mejorar.")
            break

        if fase != fase_actual:
            configurar_fase(modelo, nombre, fase)
            params = [p for p in modelo.parameters() if p.requires_grad]
            optimizador = torch.optim.AdamW(params, lr=lr_cabeza if fase == 1 else lr_red, weight_decay=1e-4)
            scheduler = None
            if fase == 2:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas_red)
                for _ in range(ep - epocas_cabeza):  # si se reanuda a mitad de la fase 2
                    scheduler.step()
                if fase_previa != 2:
                    sin_mejora = 0
            fase_actual = fase

        poner_modo_entrenamiento(modelo, nombre, fase)
        t0 = time.time()
        perdida_train = entrenar_epoca(modelo, loaders["train"], criterio, optimizador, escalador, dispositivo)
        if scheduler is not None:
            scheduler.step()

        y_val, p_val, perdida_val = predecir(modelo, loaders["val"], dispositivo, criterio)
        f1_val = f1_score(y_val, p_val, average="macro", zero_division=0)
        segundos = time.time() - t0

        historial.append(
            {
                "epoca": ep + 1,
                "fase": fase,
                "perdida_train": perdida_train,
                "perdida_val": perdida_val,
                "f1_macro_val": f1_val,
                "segundos": segundos,
            }
        )

        if f1_val > mejor_f1:
            mejor_f1, mejor_epoca, sin_mejora = f1_val, ep + 1, 0
            torch.save(modelo.state_dict(), ruta_mejor)
        else:
            sin_mejora += 1

        torch.save(
            {
                "modelo": modelo.state_dict(),
                "historial": historial,
                "mejor_f1": mejor_f1,
                "mejor_epoca": mejor_epoca,
                "sin_mejora": sin_mejora,
                "epoca": ep + 1,
                "fase": fase,
            },
            ruta_ultimo,
        )
        marca = " *" if mejor_epoca == ep + 1 else ""
        print(
            f"Epoca {ep + 1:2d}/{total_epocas} (fase {fase}) | perdida train {perdida_train:.3f} | "
            f"perdida val {perdida_val:.3f} | F1 macro val {f1_val:.3f}{marca} | {segundos:.0f} s"
        )

    modelo.load_state_dict(torch.load(ruta_mejor, map_location=dispositivo))
    return modelo, pd.DataFrame(historial)


# ---------------------------------------------------------------- metricas

def calcular_metricas(y_real, y_pred, clases):
    """Mismas metricas que el notebook de modelos clasicos."""
    etiquetas = list(range(len(clases)))
    f1_clase = f1_score(y_real, y_pred, labels=etiquetas, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_real, y_pred)),
        "f1_macro": float(f1_score(y_real, y_pred, average="macro", zero_division=0)),
        "f1_por_clase": {c: float(v) for c, v in zip(clases, f1_clase)},
    }
