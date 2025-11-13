import os
import pandas as pd
import cv2
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# === CONFIGURACIÓN ===
CSV_FILE = "rois.csv"  # archivo con columnas: imagen, x, y, w, h
IMAGES_DIR = "set_extraido"  # carpeta donde están las imágenes originales
DATASET_DIR = "dataset"
CLASS_NAME = "objeto"  # clase genérica para detección

# === 1. Leer CSV ===
df = pd.read_csv(CSV_FILE)

# Validar columnas mínimas
if not {'imagen', 'x', 'y', 'w', 'h'}.issubset(df.columns):
    raise ValueError("⚠️ El CSV debe tener las columnas: imagen, x, y, w, h")

# === 2. Crear carpetas ===
for split in ['train', 'val']:
    os.makedirs(f"{DATASET_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/labels/{split}", exist_ok=True)

# === 3. Dividir dataset (80% train / 20% val) ===
imagenes_unicas = df['imagen'].unique()
train_imgs, val_imgs = train_test_split(imagenes_unicas, test_size=0.2, random_state=42)

def get_split(img_name):
    return 'train' if img_name in train_imgs else 'val'

# === 4. Convertir a formato YOLO ===
for img_name in imagenes_unicas:
    split = get_split(img_name)
    label_path = f"{DATASET_DIR}/labels/{split}/{os.path.splitext(img_name)[0]}.txt"

    # Leer imagen para obtener dimensiones
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ No se pudo leer {img_name}, se salta.")
        continue
    h_img, w_img = img.shape[:2]

    # Filtrar las ROIs de esa imagen
    df_img = df[df['imagen'] == img_name]
    lines = []
    for _, row in df_img.iterrows():
        x_center = (row['x'] + row['w'] / 2) / w_img
        y_center = (row['y'] + row['h'] / 2) / h_img
        w_norm = row['w'] / w_img
        h_norm = row['h'] / h_img
        class_id = 0  # única clase
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Guardar el .txt de etiquetas
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    # Copiar imagen a carpeta correspondiente
    shutil.copy(img_path, f"{DATASET_DIR}/images/{split}/")

print("✅ Archivos convertidos al formato YOLO.")

# === 5. Crear archivo data.yaml ===
yaml_path = f"{DATASET_DIR}/data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: {os.path.abspath(DATASET_DIR)}/images/train\n")
    f.write(f"val: {os.path.abspath(DATASET_DIR)}/images/val\n\n")
    f.write("nc: 1\n")
    f.write(f"names: ['{CLASS_NAME}']\n")

print(f"✅ Archivo data.yaml generado en {yaml_path}")

# === 6. Entrenar YOLO ===
print("🚀 Iniciando entrenamiento YOLOv8n...")

model = YOLO("yolov8n.pt")  # modelo pequeño y rápido
model.train(
    data=yaml_path,
    epochs=30,      # más entrenamiento
    imgsz=640,      # mejor resolución
    batch=16,
    name="detector_objetos_raros",
    project="runs/train"
)

print("✅ Entrenamiento finalizado.")

