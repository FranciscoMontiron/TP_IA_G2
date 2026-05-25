# TP Cuatrimestral — Redes Neuronales Artificiales
**UTN FRLP 2026 — Grupo 2**

Clasificación de cáncer de piel con MLP (Backpropagation) usando el dataset HAM10000.
Accuracy obtenido: **64.54%** con arquitectura 784 → 512 → 256 → 128 → 7.

---

## Estructura del proyecto

```
TP IA/
├── app.py                       # Aplicación web Gradio
├── modelo_mlp.pkl               # Modelo entrenado (incluido en el repo)
├── train_and_save.py            # Script para re-entrenar el modelo
├── download_data.py             # Descarga del dataset (Drive o Kaggle)
├── skin_cancer_RNA.ipynb        # Notebook principal de entrenamiento
├── requirements.txt             # Dependencias Python
├── Dockerfile                   # Imagen Docker de la app
├── docker-compose.yml           # Orquestación Docker
├── docs/                        # Documentación y entregas
└── figures/                     # Gráficos generados por el notebook
```

---

## ⚡ Opción A — Docker (recomendado)

El modelo ya está incluido en el repo. Docker solo instala dependencias y lanza la app.

### Requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo.

### Comandos

```bash
# Primera vez
docker-compose up --build

# Veces siguientes
docker-compose up

# Detener
docker-compose down
```

Abrí el navegador en **http://localhost:7860**.

---

## 🛠️ Opción B — Instalación local

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Lanzar la app

El modelo ya está en el repo (`modelo_mlp.pkl`), así que podés correr la app directamente:

```bash
python app.py
```

Se abre automáticamente en **http://localhost:7860**.

---

## 🔁 Re-entrenar el modelo (opcional)

Si querés entrenar el modelo desde cero necesitás el dataset HAM10000.

### Descargar el dataset

```bash
python download_data.py
```

Descarga automáticamente `archive.zip` (~3 GB) desde Google Drive y lo descomprime en `archive/`.

> Descarga manual: **https://drive.google.com/file/d/1tikVuhOOlZ3-klxTUgNS_3pFykmaVXYK/view**
> Descomprimilo en `archive/` en la raíz del proyecto.

### Entrenar

```bash
python train_and_save.py
```

Genera un nuevo `modelo_mlp.pkl` en la raíz del proyecto.

---

## 📱 Funcionalidades de la app

- 📁 **Subir** una imagen de lesión cutánea
- 📷 **Capturar** con la cámara web
- 📋 **Pegar** desde el portapapeles

La app clasifica la lesión en 7 categorías:

| Nivel | Clases |
|---|---|
| 🔴 Peligroso | mel (Melanoma), bcc (Carcinoma basocelular) |
| 🟡 Consultar médico | akiec (Queratosis actínica) |
| 🟢 Benigno | bkl, df, nv, vasc |

> ⚕️ **Aviso:** No reemplaza el diagnóstico médico profesional.
