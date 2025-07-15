from django.urls import path
from . import views

urlpatterns = [
    path('classify/', views.ClassifyAPIView.as_view(), name='classify')
]
