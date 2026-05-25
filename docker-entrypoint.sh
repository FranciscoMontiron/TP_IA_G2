#!/bin/bash
# ── Entrypoint del contenedor ─────────────────────────────────────────────────
# Si no hay modelo, pregunta al usuario qué quiere hacer:
#   1) Descargar modelo pre-entrenado (~13 MB, segundos)
#   2) Descargar dataset y entrenar desde cero (~3 GB + varios minutos)
# ─────────────────────────────────────────────────────────────────────────────
set -e

MODEL_FILE="${MODEL_PATH:-/app/model/modelo_mlp.pkl}"
DATASET_FILE="/app/archive/hmnist_28_28_L.csv"
MODEL_GDRIVE_ID="1ADbyFVULbkTcFLLZtantUZO6iMvkXjxM"

echo ""
echo "======================================================"
echo "  Clasificador de Lesiones Cutáneas — UTN FRLP 2026  "
echo "======================================================"
echo ""

# ── Si el modelo ya existe, arrancar directo ──────────────────────────────────
if [ -f "$MODEL_FILE" ]; then
    echo "✅ Modelo encontrado. Lanzando app..."
    echo ""
    exec python app.py
fi

# ── No hay modelo: preguntar qué hacer ───────────────────────────────────────
echo "⚠️  No se encontró el modelo entrenado."
echo ""
echo "¿Qué querés hacer?"
echo ""
echo "  1) Descargar modelo pre-entrenado  (recomendado)"
echo "     → Solo ~13 MB · listo en segundos"
echo ""
echo "  2) Descargar dataset y entrenar desde cero"
echo "     → ~3 GB de descarga + varios minutos de entrenamiento"
echo ""

# Leer opción (con timeout de 30 s para ambientes no interactivos → default 1)
if [ -t 0 ]; then
    read -rp "Opción [1/2] (default: 1): " OPCION
else
    echo "Modo no interactivo detectado → usando opción 1 (modelo pre-entrenado)"
    OPCION="1"
fi

OPCION="${OPCION:-1}"
echo ""

# ── Opción 1: descargar modelo ────────────────────────────────────────────────
if [ "$OPCION" = "1" ]; then
    echo "📥 Descargando modelo pre-entrenado desde Google Drive..."
    mkdir -p "$(dirname "$MODEL_FILE")"
    python -c "
import gdown, sys
ok = gdown.download(id='$MODEL_GDRIVE_ID', output='$MODEL_FILE', quiet=False, fuzzy=True)
if not ok:
    print('ERROR: No se pudo descargar el modelo.')
    sys.exit(1)
"
    echo ""
    echo "✅ Modelo descargado en: $MODEL_FILE"

# ── Opción 2: descargar dataset + entrenar ────────────────────────────────────
elif [ "$OPCION" = "2" ]; then
    if [ ! -f "$DATASET_FILE" ]; then
        echo "📥 Descargando dataset desde Google Drive (~3 GB)..."
        python download_data.py
        echo ""
    else
        echo "✅ Dataset ya disponible."
    fi

    echo "🏋️  Entrenando modelo (puede tardar varios minutos)..."
    echo ""
    python train_and_save.py
    echo ""
    echo "✅ Modelo entrenado y guardado en: $MODEL_FILE"

else
    echo "❌ Opción inválida. Reiniciá el contenedor y elegí 1 o 2."
    exit 1
fi

# ── Lanzar app ────────────────────────────────────────────────────────────────
echo ""
echo "🚀 Lanzando app en http://localhost:7860 ..."
echo ""
exec python app.py
