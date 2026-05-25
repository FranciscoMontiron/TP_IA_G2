# ── Clasificador de Lesiones Cutáneas — UTN FRLP 2026 ─────────────────────────
FROM python:3.11-slim

LABEL maintainer="Grupo 2 — UTN FRLP"
LABEL description="MLP Clasificador de Lesiones Cutáneas (HAM10000)"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_LAUNCH_INBROWSER=false

WORKDIR /app

# ── Dependencias ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código fuente y modelo ────────────────────────────────────────────────────
COPY app.py .
COPY train_and_save.py .
COPY download_data.py .
COPY modelo_mlp.pkl .
COPY docker-entrypoint.sh .
RUN sed -i 's/\r$//' docker-entrypoint.sh && chmod +x docker-entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["./docker-entrypoint.sh"]
