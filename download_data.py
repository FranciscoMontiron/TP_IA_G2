"""
Descarga el dataset HAM10000 desde Kaggle y lo organiza en la carpeta archive/.

Requisitos:
    pip install kaggle
    Tener el archivo kaggle.json en ~/.kaggle/ (ver README.md)
"""

import os
import zipfile
import subprocess
import sys

DATASET = "kmader/skin-cancer-mnist-ham10000"
DEST = "archive"

def main():
    if not os.path.exists(DEST):
        os.makedirs(DEST)

    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json):
        print("ERROR: No se encontró ~/.kaggle/kaggle.json")
        print("Seguí las instrucciones del README.md para obtener tu API key de Kaggle.")
        sys.exit(1)

    print(f"Descargando dataset '{DATASET}'...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DEST],
        check=True
    )

    zip_path = os.path.join(DEST, "skin-cancer-mnist-ham10000.zip")
    if os.path.exists(zip_path):
        print("Descomprimiendo...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DEST)
        os.remove(zip_path)
        print("Listo.")
    else:
        print("No se encontró el zip descargado. Revisá si la descarga fue exitosa.")

if __name__ == "__main__":
    main()
