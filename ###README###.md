PROYECTO-PLAGA-DETECTOR
El proyecto consiste en desarrollar un sistema sencillo de visión por computadora que pueda detectar la presencia de plagas (roedores, aves o insectos grandes) en huertas o jardines domésticos. No se busca una solución industrial, sino un prototipo académico con conocimientos básicos de programación e inteligencia artificial.




#  Detector de Plagas con YOLO + Django + HuggingFace API

Este proyecto permite **entrenar un modelo YOLO personalizado** para detectar plagas u objetos en imágenes, y luego usar una **web en Django** que recibe una imagen, detecta la región de interés (ROI) y finalmente consulta una **API de HuggingFace** para clasificar la especie.

---

##  Funcionalidades principales

- Extracción masiva de imágenes desde un ZIP.
- Selección manual de ROIs para crear dataset anotado.
- Entrenamiento de un modelo YOLO personalizado.
- API de inferencia sobre imágenes.
- Web en Django donde el usuario sube una imagen y obtiene:
  - Bounding boxes generados por YOLO.
  - Clasificación de especie usando HuggingFace.

---

##  Estructura del proyecto

plaga_detector/
│── modelos/ ← modelos YOLO entrenados
│── set_extraido/ ← imágenes extraídas del ZIP
│── rois.csv ← coordenadas seleccionadas manualmente
│── extraer_imagenes.py ← script: extrae imágenes del ZIP
│── analisis2.py ← script: selecciona ROIs manualmente
│── entrenamiento2.py ← script: entrena YOLO
│── procesador.py ← usa YOLO + HuggingFace API
│── manage.py ← Django
│── app/ ← Web con Django

yaml
Copiar código

---

##  Librerías utilizadas

Asegurate de instalarlas:

```bash
pip install zipfile36
pip install pillow
pip install pandas
pip install opencv-python
pip install scikit-learn
pip install ultralytics
pip install requests
pip install django
Librerías nativas (no necesitan instalación):


 1. Extracción del dataset
El archivo extraer_imagenes.py sirve solo la primera vez.

Extrae todas las imágenes desde un ZIP gigante a la carpeta set_extraido/.

Ejecutar:

bash
Copiar código
python extraer_imagenes.py
 2. Selección manual de ROIs
El script analisis2.py permite:

✔️ Ver la imagen en color, grises y edges
✔️ Seleccionar el ROI con el mouse
✔️ Guardar la coordenada en rois.csv

Ejecutar:

bash
Copiar código
python analisis2.py
El script recorre automáticamente todas las imágenes de set_extraido/.

 3. Entrenamiento del modelo YOLO
El script entrenamiento2.py usa:

Las imágenes de set_extraido/

Las coordenadas de rois.csv

Podés ajustar:

épocas

tamaño de imagen

batch size

augmentación

Ejecutar:

bash
Copiar código
python entrenamiento2.py
El modelo entrenado se guarda en:

bash
Copiar código
modelos/mi_modelo.pt
🌐 4. Uso de la web con Django
Primero, ubicate dentro de la carpeta principal del proyecto:

bash
Copiar código
cd plaga_detector
Iniciar el servidor:

bash
Copiar código
python manage.py runserver
La web se abre en:

cpp
Copiar código
http://127.0.0.1:8000/
Allí podrás subir una imagen JPG y el sistema:

Usa YOLO → detecta el objeto

Recorta la región detectada

Envía el recorte a la API de HuggingFace

Devuelve la especie o tipo correspondiente

⚠️ Token de HuggingFace obligatorio
El archivo procesador.py usa una API privada de HuggingFace.

Tenés que reemplazar TU TOKEN en:

python
Copiar código
TOKEN = "TU_TOKEN_AQUI"
Si no lo hacés, la web no funcionará.

Para generar uno:

Ir a 👉 https://huggingface.co/settings/tokens

Crear un token con permisos "read"

Copiarlo dentro de procesador.py

▶️ Flujo completo de uso
Extraer imágenes

Seleccionar ROIs

Entrenar YOLO

Guardar el modelo entrenado en /modelos

Agregar tu token de HuggingFace

Levantar Django

Subir una imagen y analizar




