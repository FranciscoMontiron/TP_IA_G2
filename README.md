# TP Cuatrimestral — Redes Neuronales Artificiales
**UTN FRLP 2026 — Grupo 2**

Clasificación de cáncer de piel con MLP (Backpropagation) usando el dataset HAM10000.

---

## Configuración del entorno

### 1. Instalar dependencias

```bash
pip install scikit-learn pandas numpy matplotlib seaborn kaggle
```

### 2. Descargar el dataset

El dataset **HAM10000** no está incluido en el repositorio por su tamaño (~3 GB).
Hay que descargarlo desde Kaggle con los siguientes pasos:

#### a) Obtener la API key de Kaggle

1. Entrá a [kaggle.com](https://www.kaggle.com) e iniciá sesión
2. Ir a **Settings → API → Create New Token**
3. Se descarga un archivo `kaggle.json`
4. Moverlo a la carpeta `~/.kaggle/`:
   - **Windows:** `C:\Users\TuUsuario\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

#### b) Ejecutar el script de descarga

```bash
python download_data.py
```

Esto descarga y descomprime todo en la carpeta `archive/`. Al finalizar debería quedar así:

```
archive/
├── HAM10000_images_part_1/   (imágenes originales)
├── HAM10000_images_part_2/   (imágenes originales)
├── HAM10000_metadata.csv
├── hmnist_8_8_L.csv          ← usado por el notebook
├── hmnist_8_8_RGB.csv
├── hmnist_28_28_L.csv
└── hmnist_28_28_RGB.csv
```

---

## Uso

Abrí y ejecutá el notebook:

```bash
jupyter notebook skin_cancer_RNA.ipynb
```

El notebook usa `archive/hmnist_8_8_L.csv` (imágenes 8×8 píxeles, escala de grises).

---

## Estructura del proyecto

```
TP IA/
├── skin_cancer_RNA.ipynb        # Notebook principal
├── download_data.py             # Script de descarga del dataset
├── archive/                     # Dataset (no incluido en el repo)
├── curva_aprendizaje.png
├── distribucion_clases.png
├── ejemplos_patrones.png
├── matriz_confusion.png
├── resultados_validacion.csv
├── Entrega_Parcial.md
└── TPCuatrimestral_RNA_FRLP_2026.pdf
```
