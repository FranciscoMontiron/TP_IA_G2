#!/bin/bash
# ── Entrypoint del contenedor ─────────────────────────────────────────────────
# El modelo ya viene incluido en la imagen (modelo_mlp.pkl).
# Lanza la app directamente.
#
# Opcional: si querés re-entrenar desde cero dentro del contenedor:
#   docker-compose run --rm skin-cancer-app python train_and_save.py
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo ""
echo "======================================================"
echo "  Clasificador de Lesiones Cutáneas — UTN FRLP 2026  "
echo "======================================================"
echo ""
echo "✅ Modelo listo."
echo "🚀 Lanzando app en http://localhost:7860 ..."
echo ""
exec python app.py
