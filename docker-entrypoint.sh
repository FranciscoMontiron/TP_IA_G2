#!/bin/bash
# ── Entrypoint del contenedor ─────────────────────────────────────────────────
# Lógica:
#   1. Si ya existe el modelo → lanzar app directamente.
#   2. Si no existe el modelo pero sí el dataset → entrenarlo y luego lanzar.
#   3. Si no hay ni modelo ni dataset → mostrar error con instrucciones.
# ─────────────────────────────────────────────────────────────────────────────
set -e

# MODEL_PATH viene del docker-compose (env var), default al directorio de trabajo
MODEL_FILE="${MODEL_PATH:-/app/modelo_mlp.pkl}"
DATASET_FILE="/app/archive/hmnist_28_28_L.csv"

echo "======================================================"
echo "  Clasificador de Lesiones Cutáneas — UTN FRLP 2026  "
echo "======================================================"
echo ""

if [ -f "$MODEL_FILE" ]; then
    echo "✅ Modelo encontrado: $MODEL_FILE"
else
    echo "⚠️  Modelo no encontrado en: $MODEL_FILE"

    if [ -f "$DATASET_FILE" ]; then
        echo "📊 Dataset encontrado. Iniciando entrenamiento..."
        echo "   (esto puede tardar varios minutos la primera vez)"
        echo ""
        python train_and_save.py
        echo ""
        echo "✅ Modelo entrenado y guardado en: $MODEL_FILE"
    else
        echo ""
        echo "❌ ERROR: No se encontró ni el modelo ni el dataset."
        echo ""
        echo "  Ruta esperada del modelo:  $MODEL_FILE"
        echo "  Ruta esperada del dataset: $DATASET_FILE"
        echo ""
        echo "Opciones:"
        echo "  A) Si ya tenés el modelo entrenado (modelo_mlp.pkl):"
        echo "     Copialo a la carpeta del proyecto y reiniciá docker-compose."
        echo ""
        echo "  B) Si tenés el dataset en archive/:"
        echo "     Verificá que archive/hmnist_28_28_L.csv exista en el host."
        echo "     El volumen de docker-compose lo monta automáticamente."
        echo ""
        echo "  C) Para descargar el dataset desde Kaggle (fuera del contenedor):"
        echo "     1. Copiá tu kaggle.json en ~/.kaggle/"
        echo "     2. Ejecutá: python download_data.py"
        echo "     3. Luego: docker-compose up"
        exit 1
    fi
fi

echo ""
echo "🚀 Lanzando app en http://localhost:7860 ..."
echo ""
exec python app.py
