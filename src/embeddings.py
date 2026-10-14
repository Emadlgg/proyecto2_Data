"""
Extraccion de embeddings con CNN preentrenadas en ImageNet.

El preprocesamiento (normalizacion con media y desviacion de ImageNet) vive
aqui para que los notebooks y la app usen exactamente el mismo.
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESV_IMAGENET = (0.229, 0.224, 0.225)

TRANSFORMACION = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(MEDIA_IMAGENET, DESV_IMAGENET),
    ]
)

EXTRACTORES = ("resnet50", "efficientnet_b0")


class ImagenesDataset(Dataset):
    """Lee fotos de 224x224 desde una carpeta, en el orden de `archivos`."""

    def __init__(self, carpeta, archivos, transform=TRANSFORMACION):
        self.carpeta = Path(carpeta)
        self.archivos = list(archivos)
        self.transform = transform

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, i):
        im = Image.open(self.carpeta / self.archivos[i]).convert("RGB")
        return self.transform(im)


def crear_extractor(nombre, pesos=True):
    """
    Devuelve (modelo, dimension) con la cabeza de clasificacion quitada, de
    modo que la salida del modelo es el vector de caracteristicas.

    pesos=False crea el modelo con pesos aleatorios (solo para pruebas).
    """
    if nombre == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pesos else None
        modelo = models.resnet50(weights=w)
        dim = modelo.fc.in_features
        modelo.fc = torch.nn.Identity()
    elif nombre == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pesos else None
        modelo = models.efficientnet_b0(weights=w)
        dim = modelo.classifier[1].in_features
        modelo.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"Extractor desconocido: {nombre}. Opciones: {EXTRACTORES}")
    return modelo.eval(), dim


@torch.no_grad()
def extraer_embeddings(modelo, carpeta, archivos, dispositivo, batch=128, workers=2, cada=20):
    """
    Pasa todas las fotos por el modelo y devuelve un arreglo (n_fotos, dimension)
    en el mismo orden que `archivos`.
    """
    modelo = modelo.to(dispositivo)
    loader = DataLoader(
        ImagenesDataset(carpeta, archivos),
        batch_size=batch,
        shuffle=False,  # el orden tiene que coincidir con `archivos`
        num_workers=workers,
        pin_memory=(dispositivo.type == "cuda"),
    )

    salidas = []
    t0 = time.time()
    for i, lote in enumerate(loader, 1):
        lote = lote.to(dispositivo, non_blocking=True)
        with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=(dispositivo.type == "cuda")):
            emb = modelo(lote)
        salidas.append(emb.float().cpu().numpy())
        if i % cada == 0 or i == len(loader):
            hechas = min(i * batch, len(archivos))
            print(f"  {hechas}/{len(archivos)} fotos | {time.time() - t0:.0f} s")

    return np.concatenate(salidas, axis=0)
