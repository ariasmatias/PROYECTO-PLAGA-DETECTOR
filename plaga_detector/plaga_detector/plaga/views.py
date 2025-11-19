from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .procesador import procesar_imagen

fs = FileSystemStorage()

def home(request):
    return render(request, "index.html")


def procesar_imagen_view(request):
    if request.method == "POST":
        imagen = request.FILES["imagen"]

        # Guardar original
        filename = fs.save(imagen.name, imagen)
        filepath = fs.path(filename)

        # Ejecutar YOLO
        resultado = procesar_imagen(filepath)

        return render(request, "resultado.html", {
            "filename": filename,
            "processed": resultado["processed_image"],
            "detecciones": resultado["detecciones"]
        })

    return render(request, "index.html")
