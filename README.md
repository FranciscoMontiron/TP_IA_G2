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
├── download_data.py             # Descarga del dataset desde Kaggle
├── skin_cancer_RNA.ipynb        # Notebook principal de entrenamiento
├── requirements.txt             # Dependencias Python
├── Dockerfile                   # Imagen Docker de la app
├── docker-compose.yml           # Orquestación Docker
├── docker-entrypoint.sh         # Script de inicio del contenedor
├── docs/                        # Documentación y entregas
│   ├── TP_grupal_parte1_G2.pdf
│   ├── TP_grupal_parte2_G2.md
│   └── TPCuatrimestral_RNA_FRLP_2026.pdf
├── figures/                     # Gráficos generados por el notebook
│   ├── curva_aprendizaje.png
│   ├── distribucion_clases.png
│   ├── ejemplos_patrones.png
│   ├── matriz_confusion.png
│   └── resultados_validacion.csv
└── archive/                     # Dataset HAM10000 (no incluido en el repo)
    ├── hmnist_28_28_L.csv       ← usado por el notebook y train_and_save.py
    └── HAM10000_metadata.csv
```

---

## Opción A — Docker (recomendado, automatiza todo)

### Requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo.
- El dataset en `archive/hmnist_28_28_L.csv` (ver sección de descarga abajo).

### Levantar la app

```bash
# Primera vez: construye la imagen y entrena el modelo automáticamente
docker-compose up --build

# Veces siguientes: usa el modelo ya entrenado (mucho más rápido)
docker-compose up
```

Abrí el navegador en **http://localhost:7860**.

### Detener

```bash
docker-compose down
```

> **¿Cómo persiste el modelo?**
> Docker-compose usa un volumen nombrado (`model-store`) para guardar el modelo
> entre reinicios. Si querés re-entrenarlo, eliminá el volumen:
> `docker-compose down -v` y volvé a hacer `docker-compose up`.

---

## Opción B — Instalación local (manual)

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Descargar el dataset

El dataset **HAM10000** no está incluido en el repositorio por su tamaño (~3 GB).
Hay que descargarlo desde Kaggle:

#### a) Obtener la API key de Kaggle

1. Entrá a [kaggle.com](https://www.kaggle.com) e iniciá sesión.
2. Ir a **Settings → API → Create New Token**.
3. Se descarga un archivo `kaggle.json`.
4. Moverlo a:
   - **Windows:** `C:\Users\TuUsuario\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

#### b) Ejecutar el script de descarga

```bash
python download_data.py
```

Esto descarga y descomprime todo en `archive/`. Al finalizar debe quedar:

```
archive/
├── hmnist_28_28_L.csv        ← requerido por el notebook y train_and_save.py
├── hmnist_28_28_RGB.csv
├── HAM10000_metadata.csv
└── HAM10000_images_part_1/  (imágenes originales, ~5000 fotos)
└── HAM10000_images_part_2/  (imágenes originales, ~5000 fotos)
```

### 3. Entrenar el modelo

**Opción 1 — Script rápido** (sin Jupyter):
```bash
python train_and_save.py
```

**Opción 2 — Notebook completo** (con visualizaciones):
```bash
jupyter notebook skin_cancer_RNA.ipynb
```
Ejecutar todas las celdas. Los gráficos se guardan en `figures/` y el modelo como `modelo_mlp.pkl`.

### 4. Lanzar la aplicación web

```bash
python app.py
```

Se abre automáticamente en **http://localhost:7860**.

---

## Funcionalidades de la app

Desde la interfaz web podés:
- 📁 **Subir** una imagen de lesión cutánea
- 📷 **Capturar** con la cámara web
- 📋 **Pegar** desde el portapapeles

La app clasifica la lesión en 7 categorías y muestra el nivel de riesgo:

| Nivel | Clases |
|---|---|
| 🔴 Peligroso | mel (Melanoma), bcc (Carcinoma basocelular) |
| 🟡 Consultar médico | akiec (Queratosis actínica) |
| 🟢 Benigno | bkl, df, nv, vasc |

> ⚕️ **Aviso:** El modelo fue entrenado con imágenes dermatoscópicas.
> No reemplaza el diagnóstico médico profesional.

---

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `MODEL_PATH` | `modelo_mlp.pkl` | Ruta al modelo entrenado |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Dirección de escucha |
| `GRADIO_SERVER_PORT` | `7860` | Puerto de la app |
| `GRADIO_LAUNCH_INBROWSER` | `true` | Abrir browser automáticamente (desactivado en Docker) |
