"""
Descarga el dataset HAM10000 y lo organiza en la carpeta archive/.

Opciones de descarga (en orden de prioridad):
  1. Google Drive  — sin cuenta, descarga directa (recomendado)
  2. Kaggle API    — requiere cuenta y API key (ver README.md)
"""

import os
import zipfile
import subprocess
import sys

DEST = "archive"

# ── Google Drive ──────────────────────────────────────────────────────────────
# archive.zip subido por el equipo (HAM10000 completo, ~3 GB)
GDRIVE_FILE_ID  = "1tikVuhOOlZ3-klxTUgNS_3pFykmaVXYK"
GDRIVE_ZIP_NAME = "archive.zip"

# ── Kaggle (alternativa) ──────────────────────────────────────────────────────
KAGGLE_DATASET  = "kmader/skin-cancer-mnist-ham10000"


def tiene_credenciales_kaggle():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    return os.path.exists(kaggle_json)


def descargar_gdrive():
    """Descarga archive.zip desde Google Drive con gdown."""
    try:
        import gdown
    except ImportError:
        print("Instalando gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown

    zip_path = os.path.join(DEST, GDRIVE_ZIP_NAME)
    print(f"📥 Descargando dataset desde Google Drive...")
    print(f"   Destino: {zip_path}  (~3 GB, puede tardar varios minutos)")
    gdown.download(id=GDRIVE_FILE_ID, output=zip_path, quiet=False, fuzzy=True)
    return zip_path


def descargar_kaggle():
    """Descarga el dataset usando la API de Kaggle."""
    print(f"📥 Descargando dataset desde Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", DEST],
        check=True,
    )
    return os.path.join(DEST, "skin-cancer-mnist-ham10000.zip")


def descomprimir(zip_path):
    print("📦 Descomprimiendo...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DEST)
    os.remove(zip_path)
    print("✅ Dataset listo en archive/")


def main():
    os.makedirs(DEST, exist_ok=True)

    # Prioridad 1: Google Drive (sin cuenta, siempre disponible)
    print("=== Descarga del dataset HAM10000 ===")
    print()
    print("Usando Google Drive (sin necesidad de cuenta Kaggle)...")
    zip_path = descargar_gdrive()

    if not os.path.exists(zip_path):
        # Fallback: Kaggle
        if tiene_credenciales_kaggle():
            print("⚠️  Falló Google Drive. Intentando con Kaggle...")
            zip_path = descargar_kaggle()
        else:
            print("❌ No se pudo descargar el dataset.")
            print()
            print("Alternativa manual:")
            print(f"  1. Descargá archive.zip desde:")
            print(f"     https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
            print(f"  2. Descomprimilo en la carpeta '{DEST}/'")
            sys.exit(1)

    descomprimir(zip_path)


if __name__ == "__main__":
    main()
