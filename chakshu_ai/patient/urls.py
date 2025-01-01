from django.urls import path
from . import views

app_name = 'patient'

urlpatterns = [
    path('register/', views.register_patient, name='register_patient'),
    path('', views.patients_list, name='patients'),
]
