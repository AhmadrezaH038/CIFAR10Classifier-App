from django.urls import path
from . import views

app_name = 'frontend'

urlpatterns = [
    path('', views.classifier_page, name='home'),
    path('dashboard/', views.dashboard, name='dashboard')
]
