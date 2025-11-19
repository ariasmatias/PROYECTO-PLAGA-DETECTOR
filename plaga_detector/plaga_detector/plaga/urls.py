from django.urls import path
from .views import home, procesar_imagen_view

urlpatterns = [
    path('', home, name='home'),
    path('procesar/', procesar_imagen_view, name='procesar'),
]
