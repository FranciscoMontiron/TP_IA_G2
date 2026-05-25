#!/bin/bash
# ── Entrypoint del contenedor ─────────────────────────────────────────────────
# Flujo:
#   1. Modelo existe                  → lanzar app directamente.
#   2. No hay modelo, hay dataset     → entrenar → lanzar app.
#   3. No hay modelo ni dataset       → descargar dataset (Google Drive)
#                                       → entrenar → lanzar app.
# ─────────────────────────────────────────────────────────────────────────────
set -e

MODEL_FILE="${MODEL_PATH:-/app/modelo_mlp.pkl}"
DATASET_FILE="/app/archive/hmnist_28_28_L.csv"

echo "======================================================"
echo "  Clasificador de Lesiones Cutáneas — UTN FRLP 2026  "
echo "======================================================"
echo ""

# ── Paso 1: dataset ───────────────────────────────────────────────────────────
if [ ! -f "$DATASET_FILE" ]; then
    echo "⚠️  Dataset no encontrado. Descargando desde Google Drive..."
    echo "   (~3 GB, puede tardar varios minutos según la conexión)"
    echo ""
    python download_data.py
    echo ""
fi

# ── Paso 2: modelo ────────────────────────────────────────────────────────────
if [ -f "$MODEL_FILE" ]; then
    echo "✅ Modelo encontrado: $MODEL_FILE"
else
    echo "⚠️  Modelo no encontrado. Iniciando entrenamiento..."
    echo "   (esto puede tardar varios minutos)"
    echo ""
    python train_and_save.py
    echo ""
    echo "✅ Modelo guardado en: $MODEL_FILE"
fi

# ── Paso 3: lanzar app ────────────────────────────────────────────────────────
echo ""
echo "🚀 Lanzando app en http://localhost:7860 ..."
echo ""
exec python app.py
