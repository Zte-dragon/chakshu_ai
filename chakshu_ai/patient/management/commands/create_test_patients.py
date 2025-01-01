from django.core.management.base import BaseCommand
from patient.models import Patient
from django.utils import timezone
import random
from datetime import date, timedelta

class Command(BaseCommand):
    help = 'Creates 50 test patient entries'

    def handle(self, *args, **kwargs):
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'James', 'Emily', 'William', 'Olivia']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        genders = ['Male', 'Female']
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
        
        for i in range(50):
            # Generate random date of birth for ages between 20 and 80
            today = date.today()
            days_to_subtract = random.randint(365*20, 365*80)
            dob = today - timedelta(days=days_to_subtract)
            
            # Create full name
            full_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            patient = Patient.objects.create(
                name=full_name,
                date_of_birth=dob,
                gender=random.choice(genders),
                contact_number=f'+91 {random.randint(7000000000, 9999999999)}',
                address=f'{random.randint(1, 999)} {random.choice(["Main", "Park", "Lake", "Hill"])} Street, {random.choice(cities)}',
                email=f'patient{i}@example.com'
            )
            
            self.stdout.write(self.style.SUCCESS(f'Created patient: {patient.name}'))
