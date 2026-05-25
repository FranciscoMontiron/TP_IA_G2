"""
Descarga el dataset HAM10000 desde Kaggle y lo organiza en la carpeta archive/.

Credenciales (una de las dos opciones):
  A) Variables de entorno:  KAGGLE_USERNAME y KAGGLE_KEY
  B) Archivo kaggle.json en ~/.kaggle/ (ver README.md)
"""

import os
import zipfile
import subprocess
import sys

DATASET = "kmader/skin-cancer-mnist-ham10000"
DEST = "archive"


def tiene_credenciales():
    """Devuelve True si hay credenciales de Kaggle disponibles."""
    # Opción A: variables de entorno (usadas por la CLI automáticamente)
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    # Opción B: archivo kaggle.json
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    return os.path.exists(kaggle_json)


def main():
    if not tiene_credenciales():
        print("ERROR: No se encontraron credenciales de Kaggle.")
        print()
        print("  Opción A — Variables de entorno:")
        print("    KAGGLE_USERNAME=tu_usuario")
        print("    KAGGLE_KEY=tu_api_key")
        print()
        print("  Opción B — Archivo kaggle.json:")
        print("    Windows: C:\\Users\\TuUsuario\\.kaggle\\kaggle.json")
        print("    Linux/Mac: ~/.kaggle/kaggle.json")
        print()
        print("Obtené tu API key en: kaggle.com → Settings → API → Create New Token")
        sys.exit(1)

    os.makedirs(DEST, exist_ok=True)

    print(f"Descargando dataset '{DATASET}'...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DEST],
        check=True,
    )

    zip_path = os.path.join(DEST, "skin-cancer-mnist-ham10000.zip")
    if os.path.exists(zip_path):
        print("Descomprimiendo...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DEST)
        os.remove(zip_path)
        print("✅ Dataset listo en archive/")
    else:
        print("No se encontró el zip descargado. Revisá si la descarga fue exitosa.")
        sys.exit(1)


if __name__ == "__main__":
    main()
