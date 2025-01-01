from django.contrib import admin
from .models import Patient

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('name', 'date_of_birth', 'gender', 'contact_number', 'email')
    search_fields = ('name', 'email', 'contact_number')
    list_filter = ('gender',)
