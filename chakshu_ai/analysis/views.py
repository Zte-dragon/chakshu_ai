from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import json
from django.db.models import Q
from .image_processing import call_llm_api
from .models import Conversation, Message
from patient.models import Patient
import numpy as np
import base64
from io import BytesIO
from PIL import Image

@login_required
@require_http_methods(["POST"])
def new_conversation(request):
    try:
        data = json.loads(request.body)
        patient_id = data.get('patient_id')
        if not patient_id:
            return JsonResponse({'error': 'Patient ID is required'}, status=400)
        
        patient = get_object_or_404(Patient, id=patient_id)
        conversation = Conversation.objects.create(
            patient=patient,
            title=f"Analysis Session {Conversation.objects.filter(patient=patient).count() + 1}"
        )
        request.session['active_conversation'] = conversation.id
        
        return JsonResponse({
            'id': conversation.id,
            'start_time': conversation.start_time.isoformat()
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required
@require_http_methods(["POST"])
def clear_conversation(request):
    request.session.pop('active_conversation', None)
    return JsonResponse({'status': 'success'})

@login_required
def search_patients(request):
    term = request.GET.get('term', '')
    if len(term) < 2:
        return JsonResponse([], safe=False)
    
    patients = Patient.objects.filter(
        Q(name__icontains=term) | Q(id__icontains=term)
    )[:10]
    
    return JsonResponse([{
        'id': patient.id,
        'name': patient.name,
    } for patient in patients], safe=False)

@login_required
def patient_details(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    return JsonResponse({
        'id': patient.id,
        'name': patient.name,
        'age': patient.age,
        'gender': patient.gender,
        'contact_number': patient.contact_number,
    })

@login_required
def patient_conversations(request, patient_id):
    conversations = Conversation.objects.filter(patient_id=patient_id)
    return JsonResponse([{
        'id': conv.id,
        'title': conv.title,
        'start_time': conv.start_time.isoformat(),
    } for conv in conversations], safe=False)

@login_required
def get_conversation(request, conversation_id):
    conversation = get_object_or_404(Conversation, id=conversation_id)
    messages = conversation.messages.all()
    return JsonResponse([{
        'message_type': msg.message_type,
        'text': msg.text,
        'image': msg.image,
        'timestamp': msg.timestamp.isoformat(),
    } for msg in messages], safe=False)

@login_required
@require_http_methods(["GET", "POST"])
def analysis_view(request):
    if request.method == "POST":
        patient_id = request.POST.get('patient_id')
        if not patient_id:
            # Get current conversation context
            conversation_id = request.session.get('active_conversation')
            conversation_messages = []
            if conversation_id:
                try:
                    conversation = Conversation.objects.get(id=conversation_id)
                    conversation_messages = [{
                        'type': msg.message_type,
                        'text': msg.text,
                        'image': msg.image
                    } for msg in conversation.messages.all()]
                except Conversation.DoesNotExist:
                    pass

            error_message = 'Please select a patient first'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'error': error_message}, status=400)
            else:
                context = {
                    'conversation': conversation_messages,
                    'error_message': error_message,
                    'should_scroll': False
                }
                return render(request, 'analysis/analysis.html', context)
        
        patient = get_object_or_404(Patient, id=patient_id)
        conversation = None
    
        # Get active conversation or create new one
        conversation_id = request.session.get('active_conversation')
        if conversation_id:
            try:
                conversation = Conversation.objects.get(id=conversation_id)
            except Conversation.DoesNotExist:
                conversation = None
        
        if not conversation:
            conversation = Conversation.objects.create(
                patient=patient,
                title=f"Analysis Session {Conversation.objects.filter(patient=patient).count() + 1}"
            )
            request.session['active_conversation'] = conversation.id

        image = request.FILES.get('image')
        prompt = request.POST.get('prompt')
        image_b64 = None

        # Create user message
        user_message = Message(
            conversation=conversation,
            message_type='user',
            text=prompt or ""
        )

        if image:
            # Convert uploaded image to base64 for display
            image_data = image.read()
            image_b64 = base64.b64encode(image_data).decode()
            user_message.image = image_b64
            image.seek(0)

        user_message.save()

        # Get conversation history for API
        conversation_history = [{
            'type': msg.message_type,
            'text': msg.text,
            'image': msg.image
        } for msg in conversation.messages.all()]

        api_result = call_llm_api(prompt, image_b64, conversation_history)

            
            
            
        # Create assistant message
        assistant_message = Message(
            conversation=conversation,
            message_type='assistant',
            text=api_result.get('text')
        )

        # Convert processed image if exists
        if api_result.get('image') is not None:
            img = Image.fromarray(api_result['image'])
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            assistant_message.image = base64.b64encode(buffer.getvalue()).decode()

        assistant_message.save()

        # Get updated messages for display
        messages = conversation.messages.all()
        # Get patient details for repopulation
        patient_details = {
            'id': patient.id,
            'name': patient.name,
            'age': patient.age,
            'gender': patient.gender,
            'contact_number': patient.contact_number,
        }
        
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # For AJAX requests, return just the last assistant message
            last_message = messages.filter(message_type='assistant').last()
            if last_message:
                return JsonResponse({
                    'text': last_message.text,
                    'image': last_message.image
                })
            return JsonResponse({'error': 'No response generated'}, status=500)
        else:
            # For regular requests, return the full page
            context = {
                'conversation': [{
                    'type': msg.message_type,
                    'text': msg.text,
                    'image': msg.image
                } for msg in messages],
                'should_scroll': True,
                'patient_id': patient_id,
                'patient_details': patient_details
            }
            return render(request, 'analysis/analysis.html', context)

    # GET request - always start with a clean slate
    request.session.pop('active_conversation', None)
    return render(request, 'analysis/analysis.html', {
        'conversation': [],
        'should_scroll': False
    })
