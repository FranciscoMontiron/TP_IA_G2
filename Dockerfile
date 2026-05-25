# ── Clasificador de Lesiones Cutáneas — UTN FRLP 2026 ─────────────────────────
# Imagen base: Python 3.11 slim (sin extras innecesarios)
FROM python:3.11-slim

# Metadatos
LABEL maintainer="Grupo 2 — UTN FRLP"
LABEL description="MLP Clasificador de Lesiones Cutáneas (HAM10000)"

# Evitar prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Gradio: desactivar apertura automática del navegador (no tiene sentido en Docker)
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_LAUNCH_INBROWSER=false

WORKDIR /app

# ── Dependencias ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código fuente ─────────────────────────────────────────────────────────────
COPY app.py .
COPY train_and_save.py .
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# ── Puerto de la app ──────────────────────────────────────────────────────────
EXPOSE 7860

# ── Punto de entrada ──────────────────────────────────────────────────────────
ENTRYPOINT ["./docker-entrypoint.sh"]
