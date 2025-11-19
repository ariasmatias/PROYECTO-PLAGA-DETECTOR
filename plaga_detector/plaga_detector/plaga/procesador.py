# plaga/procesador.py
import cv2
import requests
from ultralytics import YOLO
from django.core.files.storage import FileSystemStorage
import uuid
import threading
import sys
import os
from typing import Any, Dict

fs = FileSystemStorage()

# ========================
# CONFIGURACIÓN
# ========================
MODEL_PATH = "C:/Users/matia/OneDrive/IFTS/Procesamiendo de imagenes v2/yolov8n.pt"
API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN = os.getenv("TOKEN")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "image/png"
}

# variable privada para cachear el modelo
_model = None
_model_lock = threading.Lock()

def get_model():
    """Carga el modelo una sola vez (thread-safe). Devuelve el objeto YOLO."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    print("🔁 Cargando modelo YOLO desde:", MODEL_PATH, file=sys.stderr)
                    _model = YOLO(MODEL_PATH)
                    print("✅ Modelo YOLO cargado correctamente.", file=sys.stderr)
                except Exception as e:
                    # no queremos que Django explote en el import; guardamos el error
                    print("❌ Error cargando modelo YOLO:", e, file=sys.stderr)
                    _model = None
    return _model

# ========================
# Clasificar recorte en HuggingFace
# ========================
def classify_crop(image_np):
    # Convertir a PNG bytes
    _, buffer = cv2.imencode(".png", image_np)
    image_bytes = buffer.tobytes()

    try:
        response = requests.post(API_URL, headers=HEADERS, data=image_bytes, timeout=20)
        if response.status_code != 200:
            return {"error": f"HF status {response.status_code}: {response.text[:200]}"}
        data = response.json()
        # data normalmente es una lista de dicts
        if isinstance(data, list) and len(data) > 0:
            best = data[0]
            return {
                "label": best.get("label", "desconocido"),
                "score": round(best.get("score", 0), 4)
            }
        return {"error": "Formato desconocido en respuesta HF", "raw": data}
    except Exception as e:
        return {"error": f"Exception sending to HF: {str(e)}"}

# ========================
# PROCESAMIENTO PRINCIPAL
# ========================
def procesar_imagen(image_path: str) -> Dict[str, Any]:
    """
    Procesa una imagen: detecta con YOLO, recorta y clasifica con HF,
    dibuja bounding boxes y guarda la imagen procesada en MEDIA.
    Retorna dict con 'detecciones' y 'processed_image' o 'error'.
    """

    # comprobar que el archivo existe
    if not os.path.exists(image_path):
        return {"error": f"No existe el archivo: {image_path}"}

    # obtener (o cargar) el modelo
    model = get_model()
    if model is None:
        return {"error": "El modelo YOLO no pudo cargarse. Revisa la ruta o los logs."}

    # leer imagen
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "No se pudo leer la imagen con OpenCV"}

    # ejecutar inferencia
    try:
        results = model(img)
    except Exception as e:
        return {"error": f"Error en inferencia YOLO: {str(e)}"}

    salida = []
    annotated_image = img.copy()

    # si no hay cajas
    try:
        boxes = results[0].boxes
    except Exception:
        boxes = []

    for i, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
        except Exception:
            # saltar detecciones inválidas
            continue

        conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
        cls_id = int(box.cls[0]) if hasattr(box, "cls") else None
        cls_name = model.names[cls_id] if (cls_id is not None and model is not None and hasattr(model, "names")) else "unknown"

        # recortar (con chequeo de límites)
        h, w = img.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w - 1, x2), min(h - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        crop = img[y1c:y2c, x1c:x2c]

        # clasificar recorte
        hf_pred = classify_crop(crop)

        # dibujar bbox y texto
        try:
            cv2.rectangle(annotated_image, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"{cls_name} {conf:.2f}",
                        (x1c, max(15, y1c - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        except Exception:
            pass

        salida.append({
            "id": i + 1,
            "bbox": [x1c, y1c, x2c, y2c],
            "clase": cls_name,
            "conf": round(conf, 3),
            "huggingface": hf_pred
        })

    # guardar imagen anotada
    processed_filename = f"processed_{uuid.uuid4().hex[:8]}.jpg"
    try:
        processed_path = fs.path(processed_filename)
        cv2.imwrite(processed_path, annotated_image)
    except Exception as e:
        return {"error": f"No se pudo guardar imagen procesada: {e}"}

    return {
        "detecciones": salida,
        "processed_image": processed_filename
    }
