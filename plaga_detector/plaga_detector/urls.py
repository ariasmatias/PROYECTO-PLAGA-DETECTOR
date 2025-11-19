from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static

urlpatterns = [
    path('', include('plaga.urls')),   # 👈 AGREGAR
    path('admin/', admin.site.urls),
]

