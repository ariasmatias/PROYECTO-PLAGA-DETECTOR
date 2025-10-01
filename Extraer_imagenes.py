import zipfile
import random
import os
from PIL import Image

# Ruta del ZIP
zip_path = "set_prueba.zip"

# Carpeta de destino
output_dir = "set_extraido"
os.makedirs(output_dir, exist_ok=True)  # Crear carpeta si no existe

# Cantidad de imágenes aleatorias a extraer
n = 50

# Abrimos el ZIP
with zipfile.ZipFile(zip_path, 'r') as z:
    # Listamos todos los archivos que sean imágenes
    all_images = [f for f in z.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Seleccionamos aleatoriamente n imágenes
    random_images = random.sample(all_images, n)

    # Extraemos y guardamos cada imagen en la carpeta de destino
    for img_name in random_images:
        with z.open(img_name) as f:
            img = Image.open(f).convert('RGB')
            # Tomamos solo el nombre del archivo (sin carpetas) para guardar
            base_name = os.path.basename(img_name)
            save_path = os.path.join(output_dir, base_name)
            img.save(save_path)

print(f"✅ Se extrajeron {n} imágenes en la carpeta '{output_dir}'")
