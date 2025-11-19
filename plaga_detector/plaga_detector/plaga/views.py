from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .procesador import procesar_imagen

fs = FileSystemStorage()


def home(request):
    return render(request, "index.html")


def procesar_imagen_view(request):
    if request.method == "POST":
        try:
            imagen = request.FILES["imagen"]

            # Guardar imagen original en /media/
            filename = fs.save(imagen.name, imagen)
            filepath = fs.path(filename)

            # Procesar imagen (YOLO + HuggingFace)
            resultado = procesar_imagen(filepath)

            # Si el procesador falló
            if not resultado or "detecciones" not in resultado:
                return render(request, "resultado.html", {
                    "error": "Ocurrió un error procesando la imagen.",
                })

            return render(request, "resultado.html", {
                "filename": filename,                                # imagen original
                "processed": resultado["processed_image"],           # imagen anotada
                "detecciones": resultado["detecciones"]             # YOLO + HF
            })

        except Exception as e:
            return render(request, "resultado.html", {
                "error": f"Error inesperado: {str(e)}"
            })

    # Mostrar home si no es POST
    return render(request, "index.html")
