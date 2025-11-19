import cv2
import requests
from ultralytics import YOLO
from PIL import Image
import io

# ========================
# CONFIGURACIÓN
# ========================
MODEL_PATH = "C:/Users/matia/OneDrive/IFTS/Procesamiendo de imagenes v2/yolov8n.pt"

# Modelo funcionando 100% con API HF
API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"

# Token (opcional para este modelo, pero lo dejamos)
TOKEN = "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"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# ========================
# Cargar modelo YOLO
# ========================
model = YOLO(MODEL_PATH)
print("Modelo YOLO cargado correctamente.")

# ========================
# Función: enviar recorte a HuggingFace
# ========================
def classify_crop(image_np):

    # Convertir recorte a PNG bytes
    _, buffer = cv2.imencode(".png", image_np)
    image_bytes = buffer.tobytes()

    headers = {
        "Content-Type": "image/png",
        "Authorization": f"Bearer {TOKEN}"
    }

    try:
        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=20)

        if response.status_code != 200:
            print("❌ Error HuggingFace:", response.status_code, response.text[:200])
            return None

        return response.json()

    except Exception as e:
        print("❌ Error enviando a HuggingFace:", str(e))
        return None


# ========================
# Función principal
# ========================
def procesar_imagen(image_path):

    print("\n📸 Procesando imagen:", image_path)

    # Cargar imagen
    img = cv2.imread(image_path)
    if img is None:
        print("No se pudo leer la imagen.")
        return

    # Ejecutar YOLO
    print("🔍 Detectando objetos...")
    results = model(img)

    # Procesar detecciones
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Recortar región detectada
        crop = img[y1:y2, x1:x2]

        # Guardar recorte localmente
        crop_filename = f"crop_{i+1}.jpg"
        cv2.imwrite(crop_filename, crop)
        print(f"🖼 Recorte guardado: {crop_filename}")

        # Enviar recorte a HuggingFace
        print(f"\n🔎 Enviando recorte {i+1} a HuggingFace...")
        response = classify_crop(crop)

        print("📌 Resultado HuggingFace:")
        print(response)

    print("\n🎉 Proceso completado.")


# ========================
# EJEMPLO DE USO
# ========================
procesar_imagen("foto_prueba.jpg")
