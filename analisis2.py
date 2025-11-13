import os
import cv2
import csv

folder = "set_extraido"
imagenes = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

csv_file = "rois.csv"

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["imagen", "x", "y", "w", "h"])  # cabecera

    for img_name in imagenes:
        path = os.path.join(folder, img_name)
        img = cv2.imread(path)

        # --- Copia para aplicar preprocesamiento ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Mostrar una vista combinada: original + filtro (opcional)
        combined = cv2.hconcat([
            img,
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        ])
        cv2.imshow("Original | Grises | Bordes", combined)
        cv2.waitKey(500)  # solo para ver antes de seleccionar

        # --- Seleccionar ROI sobre la imagen ORIGINAL ---
        roi = cv2.selectROI("Seleccionar ROI (imagen original)", img, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi
        print(f"{img_name} -> ROI seleccionada: x={x}, y={y}, w={w}, h={h}")

        # Guardar en CSV
        writer.writerow([img_name, x, y, w, h])

        # Dibujar rectángulo en la imagen original
        if w > 0 and h > 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Imagen con ROI", img)
            cv2.waitKey(500)

cv2.destroyAllWindows()
print(f"✅ Todas las coordenadas se guardaron en {csv_file}")
