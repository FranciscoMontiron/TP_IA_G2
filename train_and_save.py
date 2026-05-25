"""
Entrena el modelo MLP y guarda modelo_mlp.pkl
Equivalente a ejecutar el notebook completo.

Uso:
    python train_and_save.py

En Docker la ruta de salida se sobreescribe con la variable de entorno MODEL_PATH.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# ── Rutas ─────────────────────────────────────────────────────────────────────
# Buscamos el dataset relativo al script (funciona desde cualquier directorio)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "archive", "hmnist_28_28_L.csv")

# MODEL_PATH se puede sobreescribir vía variable de entorno (usado en Docker)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "modelo_mlp.pkl"),
)

# Crear directorio del modelo si no existe (necesario cuando MODEL_PATH apunta a un subdirectorio)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ── Carga del dataset ─────────────────────────────────────────────────────────
print("Cargando dataset...")
if not os.path.exists(DATASET):
    raise FileNotFoundError(
        f"No se encontró el dataset en '{DATASET}'.\n"
        "Ejecutá primero: python download_data.py"
    )

df = pd.read_csv(DATASET)
feature_cols = [c for c in df.columns if c.startswith("pixel")]

X = df[feature_cols].values / 255.0
y = df["label"].values

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Entrenamiento: {len(X_train):,} | Validacion: {len(X_val):,}")
print("Aplicando RandomOverSampler...")

ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
print(f"Entrenamiento balanceado: {len(X_train_res):,} muestras")

# ── Entrenamiento ─────────────────────────────────────────────────────────────
print("Entrenando MLP (puede tardar unos minutos)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=25,
)

mlp.fit(X_train_res, y_train_res)

# ── Evaluacion ────────────────────────────────────────────────────────────────
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_val, mlp.predict(X_val))
print(f"\nAccuracy en validacion: {acc*100:.2f}%")
print(f"Epocas: {mlp.n_iter_} | Loss: {mlp.loss_:.6f}")

# ── Guardado ──────────────────────────────────────────────────────────────────
joblib.dump(mlp, MODEL_PATH)
print(f"\nModelo guardado en {MODEL_PATH}")
print("Ya podes correr: python app.py")
