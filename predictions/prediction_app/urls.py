from django.urls import path
from . import views

urlpatterns = [
    path('', views.ImagePredictionView.as_view(), name='image_prediction'),
    path('info', views.InfoView.as_view(), name='info'),
    path('predictCancer/', views.ImagePredictionCanView.as_view(), name='predictCancer'),
]
