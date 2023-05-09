from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('prediction/', include('prediction_app.urls')),
    path('', include('prediction_app.urls')),
]
