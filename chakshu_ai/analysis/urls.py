from django.urls import path
from . import views

app_name = 'analysis'

urlpatterns = [
    path('', views.analysis_view, name='analysis'),
    path('search_patients/', views.search_patients, name='search_patients'),
    path('patient_details/<int:patient_id>/', views.patient_details, name='patient_details'),
    path('patient_conversations/<int:patient_id>/', views.patient_conversations, name='patient_conversations'),
    path('conversation/<int:conversation_id>/', views.get_conversation, name='get_conversation'),
    path('new_conversation/', views.new_conversation, name='new_conversation'),
    path('clear_conversation/', views.clear_conversation, name='clear_conversation'),
]
