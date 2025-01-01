from .model.models import RetinaSegmentationModel, GeneralPathologyModel, RetinalVesselSegmentation
from anthropic import Anthropic
import requests
import base64
from io import BytesIO
from PIL import Image
import re

PATHOLOGIES = [
    "diabetic retinopathy", "hemorrhages", "microaneurysms", "exudates", "hard exudates", 
    "cotton wool spots", "venous beading", "intraretinal microvascular abnormalities", 
    "neovascularization", "macular edema", "macula", "soft exudates", "drusen", 
    "age-related macular degeneration", "media haze", "vitreous", "pathologic myopia", 
    "branch retinal vein occlusion", "tessellation", "epiretinal membrane", "laser scar", 
    "central serous retinopathy", "asteroid hyalosis", "optic disc pallor", "shunt", 
    "macular hole", "retinitis pigmentosa", "glaucoma", "optic nerve", "optic cup", 
    "optic disc", "hypertensive retinopathy", "arteriolar narrowing", "vascular wall", 
    "myopic maculopathy", "cataract", "lens", "retinal detachment", "optic atrophy", 
    "myelinated nerve fibers", "choroidal rupture", "choroidal folds", "choroid", 
    "vitreous hemorrhage", "macroaneurysm", "vasculitis", "plaque", "pigment epithelial detachment", 
    "collateral vessels", "dragged disc", "congenital disc abnormality", "bietti crystalline dystrophy", 
    "peripheral retinal degeneration", "retinal break", "neoplasm", "fibrosis", "silicon oil", 
    "anterior ischemic optic neuropathy", "parafoveal telangiectasia", "retinal traction", 
    "chorioretinitis", "fundus", "retina", "fovea", "arteries", "veins", "capillaries", 
    "retinal pigment epithelium"
]

MODEL_MAP = {
    "general pathology": 0,
    "od-oc segmentation": 1,
    "retinal blood vessel segmentation": 2
}

def format_conversation(conversation):
    """Format conversation history for LLM context"""
    if not conversation:
        return ""
    formatted = []
    for msg in conversation:
        msg_type = "User" if msg['type'] == 'user' else "Assistant"
        text = msg.get('text', '')
        has_image = 'image' in msg and msg['image'] is not None
        formatted.append(f"{msg_type}: {text} {'[Image provided]' if has_image else ''}")
    return "\n".join(formatted)

def process_image_only(image_data, use_claude=False):
    """Handle case when only image is provided without prompt"""
    system_prompt = f"""You are a friendly AI assistant specialized in eye pathologies and retinal image analysis.
    Since the user has not provided any specific query, provide a welcoming response that:
    1. Acknowledges that no specific question was asked
    2. Briefly explains the system's capabilities:
       - General pathology detection for various eye conditions
       - Optic disc and cup segmentation
       - Retinal blood vessel analysis
    3. Encourages the user to ask a specific question about the image
    
    The system can detect these pathologies: {', '.join(PATHOLOGIES)}
    """
    
    welcome_prompt = "Generate a welcoming message for a new user who has uploaded an image but hasn't asked a specific question."
    
    if use_claude:
        response = llm_call_claude(system_prompt, welcome_prompt)
    else:
        response = llm_call(system_prompt, welcome_prompt)
    
    return {"text": response, "image": None}

def call_llm_api(prompt, image_data=None, conversation=None, use_claude=False):
    # Handle case where no prompt is provided (just image)
    if not prompt and image_data:
        return process_image_only(image_data, use_claude)
    


    # First LLM call to determine if query is about previous conversation or needs model analysis
    system_prompt_1 = f"""You are an AI assistant specialized in eye pathologies and retinal image analysis.
    
    Previous conversation context:
    {format_conversation(conversation) if conversation else "No previous conversation"}
    
    Your task is to:
    1. Determine if the query is:
       a) A question about previous conversation/analysis
       b) A request for new image analysis
    
    2. If it's a question about previous conversation:
       - Respond with:
       query_type: conversation
       response: [Your direct response to the question based on conversation history]
    
    3. If it's a request for new image analysis:
       - Determine which model should be used
       - Identify specific pathologies or entities mentioned
       - Respond with:
       query_type: analysis
       model_chosen: [model name]
       entities_requested: [comma-separated list of entities or 'ALL' if a general assessment is requested]

    For analysis requests:
    - Use 'general pathology' for queries about eye conditions, diseases, or specific pathologies, including abnormalities of the optic disc or cup
    - Use 'od-oc segmentation' ONLY if the query explicitly requests visualization or segmentation of the optic disc and cup
    - Use 'retinal blood vessel segmentation' for queries about blood vessel analysis
    
    The list of known pathologies includes: {', '.join(PATHOLOGIES)}
    
    Examples:
    1. Query: "What did you find about the optic disc in the previous analysis?"
       Response:
       query_type: conversation
       response: Based on the previous analysis, I'll summarize the findings about the optic disc...
    
    2. Query: "Is there abnormality in optical disc or cup?"
       Response:
       query_type: analysis
       model_chosen: general pathology
       entities_requested: abnormal optic disc, abnormal optic cup
    
    3. Query: "Can you visualize the optic disc and cup in this image?"
       Response:
       query_type: analysis
       model_chosen: od-oc segmentation
       entities_requested: None
    4. Query: "What are the blood vessels like in this image?"
         Response:
            query_type: analysis
            model_chosen: retinal blood vessel segmentation
            entities_requested: None
    5. Query: "Can you segment the oc and od?"
       Response:
       query_type: analysis
       model_chosen: od-oc segmentation
       entities_requested: None
    
    6. Query: "What all can you do?"
       Response:
       query_type: conversation
       response: I am capable of ...
    
    Doctor's query: {prompt}
    """
    
    if use_claude:
        response_1 = llm_call_claude(system_prompt_1, prompt)
    else:
        response_1 = llm_call(system_prompt_1, prompt)
    
    parsed_response = parse_llm_response(response_1)
    print(f"Parsed response:{parsed_response}")
    
    # If it's a conversation query, return the direct response
    if parsed_response.get('query_type') == 'conversation':
        return {
            "text": parsed_response.get('response', "I couldn't understand the query about the previous conversation."),
            "image": None
        }

    # If no image is provided, check the conversation history for the last image
    if not image_data:
        last_image = None
        for msg in reversed(conversation):
            if msg.get('type') == 'user'and msg.get('image'):
                last_image = msg['image']
                break
        image_data = last_image if last_image else None

    if not image_data:
        return {"text": "I'm sorry, but I couldn't find any image to analyze. Please upload an image for analysis.",
                "image": None}
    
    # Otherwise proceed with model-based analysis
    model_id = MODEL_MAP.get(parsed_response.get('model_chosen', '').lower(), -1)
    processed_image = None
    
    if model_id == 0:  # General Pathology
        entities_requested = parsed_response['entities_requested'].split(',') if parsed_response['entities_requested'] != 'ALL' else 'ALL'
        result = process_general_pathology(image_data, ai_assistant_mode=True, requested_entities=entities_requested)
        
    elif model_id == 1:  # OD-OC Segmentation
        result, processed_image = process_od_oc_segmentation(image_data)
        
    elif model_id == 2:  # Retinal Blood Vessel Segmentation
        processed_image, result = process_retinal_vessel_segmentation(image_data)
      
    else:
        return {"text": "I'm sorry, but I couldn't determine the appropriate model for your query. Please try rephrasing your question.",
                "image": None}

    # Second LLM call to generate the final response
    system_prompt_2 = f"""You are an AI assistant specialized in interpreting eye pathology results for doctors.
    
    Previous conversation context:
    {format_conversation(conversation) if conversation else "No previous conversation"}
    
    Provide a brief and direct response to the doctor's query based on the analysis results.
    Focus on the entities mentioned in the query or detected in the analysis.
    
    For general pathology:
    - For each entity, a score close to 1.0 for "X is present" indicates a high probability of the entity being present.
    - A score close to 1.0 for "X is not present" indicates a high probability of the entity being absent.
    - If an entity was asked about but not detected, explicitly state that it was not found.
    - For general assessments, summarize the most significant findings, focusing on pathologies with high probabilities of being present.

    For OD-OC segmentation, interpret the CDR and RDR values.

    Keep your response concise, ideally within 2-3 sentences for specific queries and 3-5 sentences for general assessments.

    Doctor's query: {prompt}
    Analysis results: {result}
    Requested entities: {parsed_response['entities_requested']}
    """
    
    if use_claude:
        final_response = llm_call_claude(system_prompt_2, "Generate a concise response to the doctor's query based on the analysis results.")
    else:
        final_response = llm_call(system_prompt_2, "Generate a concise response to the doctor's query based on the analysis results.")

    return {
        "text": final_response,
        "image": processed_image
    }

def llm_call_claude(system_prompt, user_prompt):
    """Make an API call to Anthropic's Claude 3 Haiku model"""
    try:
        client = Anthropic()
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        
        response_dict = {
            "choices": [
                {
                    "message": {
                        "content": message.content[0].text
                    }
                }
            ]
        }
        
        return response_dict['choices'][0]['message']['content']
        
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"

def llm_call(system_prompt, user_prompt):
    """Make an API call to the local LLM server"""
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "model": "gemma-2-2b-it",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

def parse_llm_response(response):
    """Parse LLM response into key-value pairs"""
    parsed = {}
    
    # Check if it's a conversation type response
    if 'query_type: conversation' in response:
        parsed['query_type'] = 'conversation'
        # Find the start of the response content
        response_start = response.find('response:')
        if response_start != -1:
            # Get everything after "response:" until the end
            response_text = response[response_start + 9:].strip()
            parsed['response'] = response_text
            return parsed
    
    # If not a conversation type, parse as key-value pairs
    lines = response.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            parsed[key.strip()] = value.strip()
    
    return parsed

def process_general_pathology(image_data, ai_assistant_mode=False, requested_entities=None):
    """Process image for general pathology detection"""
    model = GeneralPathologyModel()
    _, logs = model.process_image(image_data, ai_assistant_mode=ai_assistant_mode, requested_entities=requested_entities)
    return logs

def process_retinal_vessel_segmentation(image_data):
    """Process image for retinal vessel segmentation"""
    model = RetinalVesselSegmentation()
    segmented_image, logs = model.process_image(image_data)
    return segmented_image, logs
    
def process_od_oc_segmentation(image_data):
    """Process image for optic disc-cup segmentation"""
    model = RetinaSegmentationModel()
    processed_image, logs = model.process_image(image_data)
    return logs, processed_image
def formulate_response(prompt, result, model_chosen, pathologies_requested):
    if model_chosen == "general pathology":
        detected_pathologies = [path for path in PATHOLOGIES if path.lower() in result.lower()]
        requested_pathologies = [p.strip().lower() for p in pathologies_requested.split(',') if p.strip() != 'None']
        
        response = "Based on the analysis:\n"
        
        # First, address the specifically requested pathologies
        if requested_pathologies:
            response += "Regarding your specific query:\n"
            for path in requested_pathologies:
                if path in [p.lower() for p in detected_pathologies]:
                    response += f"- {path.capitalize()} was detected (probability: {extract_probability(result, path)})\n"
                else:
                    response += f"- {path.capitalize()} was not detected\n"
            response += "\n"
        
        # Then, report all detected pathologies
        if detected_pathologies:
            response += "All detected pathologies:\n"
            for path in detected_pathologies:
                response += f"- {path.capitalize()} (probability: {extract_probability(result, path)})\n"
        else:
            response += "No pathologies were detected. The retina appears to be normal.\n"
        
        # Add a note for pathologies not in the predefined list
        unknown_pathologies = [p for p in requested_pathologies if p not in [path.lower() for path in PATHOLOGIES]]
        if unknown_pathologies:
            response += "\nNote: The following requested pathologies are not in our predefined list and may not be detectable by our current model:\n"
            for path in unknown_pathologies:
                response += f"- {path.capitalize()}\n"
    
    elif model_chosen == "od-oc segmentation":
        response = f"OD-OC Segmentation results:\n{result}"
    
    else:
        response = result

    return response

def extract_probability(result, pathology):
    """Extract probability value for a pathology from result string"""
    match = re.search(f"{pathology}: (0\.\d+)", result, re.IGNORECASE)
    return match.group(1) if match else "N/A"

def resize_image(image, max_size=(300, 300)):
    """Resize image while maintaining aspect ratio"""
    image.thumbnail(max_size)
    return image


def image_upload_and_diagnosis(image, user_input):
    user_input = st.text_input("Ask a question about the diagnosis:")
    image_data = image                
    current_llm_response = call_llm_api(user_input, image_data)

    if st.session_state.current_llm_response:
        st.text_area("ChákṣuAI Response:", value=st.session_state.current_llm_response, height=200)
        
        if hasattr(st.session_state, 'processed_image') and st.session_state.processed_image is not None:
            st.subheader("Processed Image")
            st.write("Click on the image to zoom")
            display_pil_image = PILImage.fromarray(st.session_state.processed_image)
            image_zoom(display_pil_image)

    with tab2:
        st.subheader("Detailed Diagnosis")
        diagnosis_type = st.selectbox("Select Diagnosis:", [
            "Common Pathology Identification",
            "OD-OC Segmentation",
            "Blood Vessel Segmentation"
        ])
        
        if st.button(f"Process Image ({diagnosis_type})"):
            with st.spinner("Processing image..."):
                image_data = BytesIO()
                st.session_state.uploaded_image.save(image_data, format='PNG')
                image_bytes = image_data.getvalue()

                if diagnosis_type == "Common Pathology Identification":
                    model = GeneralPathologyModel()
                    _, logs = model.process_image(image_bytes)
                    st.session_state.processed_image = None
                    st.session_state.display_processed = None
                elif diagnosis_type == "OD-OC Segmentation":
                    model = RetinaSegmentationModel()
                    processed_image, logs = model.process_image(image_bytes)
                    processed_pil = PILImage.fromarray(processed_image)
                    display_processed = processed_pil.copy()
                    display_processed.thumbnail((512, 512))
                    st.session_state.processed_image = processed_image
                    st.session_state.display_processed = np.array(display_processed)
                elif diagnosis_type == "Blood Vessel Segmentation":
                    segmented_image, logs = process_retinal_vessel_segmentation(image_bytes)
                    segmented_pil = PILImage.fromarray(segmented_image)
                    display_segmented = segmented_pil.copy()
                    display_segmented.thumbnail((512, 512))
                    st.session_state.processed_image = segmented_image
                    st.session_state.display_processed = np.array(display_segmented)

                st.session_state.model_logs = logs
                st.session_state.current_diagnosis_type = diagnosis_type
                st.session_state.processing_done = True

            st.subheader(f"{diagnosis_type} Results")
            st.text(logs)

            if st.session_state.processed_image is not None:
                st.subheader("Processed Image")
                st.write("Click on the image to zoom")
                if hasattr(st.session_state, 'display_processed'):
                    display_pil_image = PILImage.fromarray(st.session_state.display_processed)
                    image_zoom(display_pil_image)

    # Save diagnosis button (outside both tabs)
    if hasattr(st.session_state, 'processing_done') and st.session_state.processing_done:
        if st.button("Save Diagnosis", key="save_diagnosis_button"):
            save_current_diagnosis(selected_patient_id)
            st.success("Diagnosis saved successfully!")
            st.session_state.processing_done = False
            st.session_state.current_llm_response = None
            if 'previous_audio_length' in st.session_state:
                del st.session_state.previous_audio_length
