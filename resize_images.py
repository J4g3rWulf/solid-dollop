from PIL import Image, ImageOps
import os

DATA_DIR = "images/train"
TARGET_SIZE = (299, 299) 

for root, dirs, files in os.walk(DATA_DIR):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(root, filename)
            try:
                img = Image.open(filepath)

                img = ImageOps.exif_transpose(img)

                img = img.convert("RGB")

                img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)

                img_padded = ImageOps.pad(img, TARGET_SIZE, color="white")

                # Sobreescreve arquivo original
                img_padded.save(filepath, quality=90)
                print(f"Processando: {filepath}")
            except Exception as e:
                print(f"Erro processsando {filepath}: {e}")

