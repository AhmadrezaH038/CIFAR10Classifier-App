from django.urls import path
from . import views

urlpatterns = [
    path('classify/', views.run_classifier, name='classify')
]
