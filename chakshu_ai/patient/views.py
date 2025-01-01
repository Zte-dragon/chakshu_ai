from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import PatientRegistrationForm
from .models import Patient

@login_required
def register_patient(request):
    if request.method == 'POST':
        form = PatientRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('patient:patients')
    else:
        form = PatientRegistrationForm()
    return render(request, 'patient/register_patient.html', {'form': form})

@login_required
def patients_list(request):
    patients = Patient.objects.all().order_by('-id')
    return render(request, 'patient/patients.html', {'patients': patients})
