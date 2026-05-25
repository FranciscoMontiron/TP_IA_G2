#!/bin/bash
# ── Entrypoint del contenedor ─────────────────────────────────────────────────
# Flujo:
#   1. Modelo existe                              → lanzar app.
#   2. No hay modelo, pero hay dataset            → entrenar → lanzar app.
#   3. No hay modelo ni dataset, hay creds Kaggle → descargar → entrenar → lanzar.
#   4. Nada de lo anterior                        → error con instrucciones.
# ─────────────────────────────────────────────────────────────────────────────
set -e

MODEL_FILE="${MODEL_PATH:-/app/modelo_mlp.pkl}"
DATASET_FILE="/app/archive/hmnist_28_28_L.csv"

echo "======================================================"
echo "  Clasificador de Lesiones Cutáneas — UTN FRLP 2026  "
echo "======================================================"
echo ""

# ── Paso 1: verificar o descargar dataset ────────────────────────────────────
if [ ! -f "$DATASET_FILE" ]; then
    echo "⚠️  Dataset no encontrado en: $DATASET_FILE"

    if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
        echo "📥 Credenciales de Kaggle detectadas. Descargando dataset..."
        echo "   (esto puede tardar varios minutos según la conexión)"
        python download_data.py
        echo ""
    else
        echo ""
        echo "❌ No hay dataset ni credenciales de Kaggle."
        echo ""
        echo "  Opción A — Descarga automática (pasar credenciales al contenedor):"
        echo "    En docker-compose.yml, descomentar y completar:"
        echo "      KAGGLE_USERNAME: tu_usuario"
        echo "      KAGGLE_KEY: tu_api_key"
        echo "    Luego: docker-compose up"
        echo ""
        echo "  Opción B — Descarga manual (fuera del contenedor):"
        echo "    1. Copiá kaggle.json en ~/.kaggle/"
        echo "    2. python download_data.py"
        echo "    3. docker-compose up"
        exit 1
    fi
fi

# ── Paso 2: verificar o entrenar modelo ──────────────────────────────────────
if [ -f "$MODEL_FILE" ]; then
    echo "✅ Modelo encontrado: $MODEL_FILE"
else
    echo "⚠️  Modelo no encontrado. Iniciando entrenamiento..."
    echo "   (esto puede tardar varios minutos la primera vez)"
    echo ""
    python train_and_save.py
    echo ""
    echo "✅ Modelo entrenado y guardado en: $MODEL_FILE"
fi

# ── Paso 3: lanzar app ────────────────────────────────────────────────────────
echo ""
echo "🚀 Lanzando app en http://localhost:7860 ..."
echo ""
exec python app.py
