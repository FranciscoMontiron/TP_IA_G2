# TP Cuatrimestral — Redes Neuronales Artificiales
**UTN FRLP 2026 — Grupo 2**

Clasificación de cáncer de piel con MLP (Backpropagation) usando el dataset HAM10000.
Accuracy obtenido: **64.54%** con arquitectura 784 → 512 → 256 → 128 → 7.

---

## Estructura del proyecto

```
TP IA/
├── app.py                       # Aplicación web Gradio
├── train_and_save.py            # Script de entrenamiento standalone
├── download_data.py             # Descarga del dataset (Drive o Kaggle)
├── skin_cancer_RNA.ipynb        # Notebook principal de entrenamiento
├── requirements.txt             # Dependencias Python
├── Dockerfile                   # Imagen Docker de la app
├── docker-compose.yml           # Orquestación Docker
├── docker-entrypoint.sh         # Script de inicio del contenedor
├── docs/                        # Documentación y entregas
├── figures/                     # Gráficos generados por el notebook
└── archive/                     # Dataset HAM10000 (no incluido en el repo)
    └── hmnist_28_28_L.csv       ← requerido para entrenar
```

---

## ⚡ Opción A — Docker (recomendado, automatiza todo)

Con Docker no necesitás instalar nada manualmente. El contenedor se encarga solo de descargar el dataset, entrenar el modelo y lanzar la app.

### Requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo.

### Levantar la app

```bash
# Primera vez: construye la imagen, descarga dataset (~3 GB), entrena y lanza
docker-compose up --build

# Veces siguientes: el modelo ya está guardado, arranca directo
docker-compose up
```

Abrí el navegador en **http://localhost:7860**.

```bash
# Detener
docker-compose down

# Detener y borrar el modelo guardado (para re-entrenar desde cero)
docker-compose down -v
```

---

## 🛠️ Opción B — Instalación local

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Obtener el dataset

**Opción rápida — Google Drive (sin cuenta Kaggle):**

```bash
python download_data.py
```

Esto descarga automáticamente el dataset desde Google Drive (~3 GB) y lo descomprime en `archive/`.

> Si preferís descargarlo manualmente:
> 1. Descargá `archive.zip` desde este link:
>    **https://drive.google.com/file/d/1tikVuhOOlZ3-klxTUgNS_3pFykmaVXYK/view**
> 2. Descomprimilo en la raíz del proyecto. Debe quedar así:
>    ```
>    archive/
>    ├── hmnist_28_28_L.csv     ← este es el que usa el modelo
>    ├── hmnist_28_28_RGB.csv
>    └── HAM10000_metadata.csv
>    ```

**Opción alternativa — Kaggle API:**

1. Ir a [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token
2. Mover el `kaggle.json` descargado a `C:\Users\TuUsuario\.kaggle\` (Windows) o `~/.kaggle/` (Linux/Mac)
3. Ejecutar: `python download_data.py`

### 3. Entrenar el modelo

**Script rápido** (sin Jupyter):
```bash
python train_and_save.py
```
Genera `modelo_mlp.pkl` en la raíz del proyecto.

**Notebook completo** (con visualizaciones y análisis):
```bash
jupyter notebook skin_cancer_RNA.ipynb
```
Ejecutar todas las celdas. Los gráficos se guardan en `figures/`.

### 4. Lanzar la aplicación

```bash
python app.py
```

Se abre automáticamente en **http://localhost:7860**.

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

---

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `MODEL_PATH` | `modelo_mlp.pkl` | Ruta al modelo entrenado |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Dirección de escucha |
| `GRADIO_SERVER_PORT` | `7860` | Puerto de la app |
| `GRADIO_LAUNCH_INBROWSER` | `true` | Abrir navegador automáticamente (desactivado en Docker) |
| `KAGGLE_USERNAME` | — | Usuario Kaggle (alternativa a kaggle.json) |
| `KAGGLE_KEY` | — | API key Kaggle (alternativa a kaggle.json) |
