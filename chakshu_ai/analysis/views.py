from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from .image_processing import call_llm_api
import numpy as np
import base64
from io import BytesIO
from PIL import Image

@login_required
@require_http_methods(["GET", "POST"])
def analysis_view(request):
    # Initialize conversation if it doesn't exist
    if request.method == "GET":
        request.session['conversation'] = []
        request.session.modified = True

    conversation = request.session.get('conversation', [])
    
    if request.method == "POST":
        image = request.FILES.get('image')
        prompt = request.POST.get('prompt')
        image_b64 = None
        user_message = {
                'type': 'user',
                'text': "",
                'image': None
            }
        
        if prompt:  # Allow messages without images
            # Create user message
            user_message['text'] = prompt
        # Process image if provided
        if image:
            # Convert uploaded image to base64 for display
            image_data = image.read()
            image_b64 = base64.b64encode(image_data).decode()
            user_message['image'] = image_b64
            
            # Reset file pointer for processing
            image.seek(0)

        api_result = call_llm_api(prompt, image_b64, conversation)

        # # Handle different use cases
        # # 1. Image and prompt
        # # 2. Image and no prompt (welcome message)
        # # 3. No image and prompt (follow-up question using previous image)
        # if image_b64 and prompt:
        #     # Case 1: Both image and prompt
        #     api_result = call_llm_api(prompt, image_b64, conversation)
        # elif image_b64 and not prompt:
        #     # Case 2: Only image - show welcome message
        #     api_result = call_llm_api(None, image_b64, conversation)
        # elif prompt and not image_b64:
        #     # Case 3: Only prompt - use last image from conversation if available
            
        # else:
        #     # No image and no prompt - shouldn't happen due to form validation
        #     api_result = {"text": "Please provide an image or a question.", "image": None}
            
            
            
        # Create assistant message
        assistant_message = {
            'type': 'assistant',
            'text': api_result.get('text'),
            'image': None
        }
            
        # Convert processed image if exists
        if api_result.get('image') is not None:
            img = Image.fromarray(api_result['image'])
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            assistant_message['image'] =   base64.b64encode(buffer.getvalue()).decode()
        
        # Update conversation history
        conversation.extend([user_message, assistant_message])
        request.session['conversation'] = conversation
        request.session.modified = True
        
        # Force session save
        request.session.save()
        
        context = {
            'conversation': conversation,
            'last_result': assistant_message,
            'should_scroll': True
        }
        return render(request, 'analysis/analysis.html', context)
    
    context = {
        'conversation': conversation,
        'should_scroll': False
    }
    return render(request, 'analysis/analysis.html', context)
