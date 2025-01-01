import streamlit as st
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportLabImage
import io
import numpy as np
from streamlit_image_zoom import image_zoom
from models import RetinaSegmentationModel, GeneralPathologyModel, RetinalVesselSegmentation
import re
import uuid
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import csv
import os
import requests
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
import tempfile
import shutil
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import librosa
import soundfile as sf
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from audiorecorder import audiorecorder
from transformers import pipeline
from streamlit import cache_resource
from pydub import AudioSegment
    
st.set_page_config(page_title="Chaksu AI", layout="wide")
# CSV files for user and patient storage
USER_DB = 'users.csv'
PATIENT_DB = 'patients.csv'
DIAGNOSIS_DB = 'diagnoses.csv'
IMAGE_FOLDER = 'images'
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
CONVERSATION_DB = 'conversations.csv'

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

def ensure_recordings_folder():
    if not os.path.exists('recordings'):
        os.makedirs('recordings')

def save_image(image, patient_id, image_type):
    """Save image to file system"""
    filename = f"{image_type}_{patient_id}_{uuid.uuid4()}.png"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    image.save(filepath)
    return filepath

def load_image(filepath):
    """Load image from file system"""
    return PILImage.open(filepath)
    
def call_llm_api(prompt, image_data=None):
    # First LLM call to determine the model and entities
    system_prompt_1 = f"""You are an AI assistant specialized in eye pathologies and retinal image analysis. 
    Your task is to determine which model should be used based on the user's query and identify any specific pathologies or entities mentioned or implied.
    Respond only with the following format:
    model_chosen: [model name]
    entities_requested: [comma-separated list of entities or 'ALL' if a general assessment is requested]

    Use 'general pathology' for queries about eye conditions, diseases, or specific pathologies, including abnormalities of the optic disc or cup.
    Use 'od-oc segmentation' ONLY if the query explicitly requests visualization or segmentation of the optic disc and cup.
    Use 'retinal blood vessel segmentation' for queries about blood vessel analysis.

    Important: If the query mentions abnormalities or conditions related to the optic disc or cup without explicitly requesting segmentation or visualization, use 'general pathology'.

    The list of known pathologies includes: {', '.join(PATHOLOGIES)}
    If the query mentions or implies any of these pathologies or other entities, include them in the entities_requested field.
    If the doctor asks about an entity not in this list, include it in the entities_requested field as well.
    If the doctor asks for a general assessment of pathologies, use 'ALL' as the entities_requested value.

    Examples:
    1. Query: "Is there abnormality in optical disc or cup?"
       Response:
       model_chosen: general pathology
       entities_requested: abnormal optic disc, abnormal optic cup

    2. Query: "Can you visualize the optic disc and cup in this image?"
       Response:
       model_chosen: od-oc segmentation
       entities_requested: None

    3. Query: "Perform OD-OC segmentation on this retinal image."
       Response:
       model_chosen: od-oc segmentation
       entities_requested: None

    Doctor's query: {prompt}
    """

    response_1 = llm_call(system_prompt_1, prompt)
    parsed_response = parse_llm_response(response_1)
    
    model_id = MODEL_MAP.get(parsed_response['model_chosen'].lower(), -1)
    
    if model_id == 0:  # General Pathology
        entities_requested = parsed_response['entities_requested'].split(',') if parsed_response['entities_requested'] != 'ALL' else 'ALL'
        result = process_general_pathology(image_data, ai_assistant_mode=True, requested_entities=entities_requested)
        st.session_state.processed_image = None
        st.session_state.model_logs = result
    elif model_id == 1:  # OD-OC Segmentation
        result, processed_image = process_od_oc_segmentation(image_data)
        st.session_state.processed_image = processed_image
        st.session_state.model_logs = result
    elif model_id == 2:  # Retinal Blood Vessel Segmentation
        processed_image, result = process_retinal_vessel_segmentation(image_data)
        st.session_state.processed_image = processed_image
        st.session_state.model_logs = result
    else:
        return "I'm sorry, but I couldn't determine the appropriate model for your query. Please try rephrasing your question."

    # Second LLM call to generate the final response
    system_prompt_2 = f"""You are an AI assistant specialized in interpreting eye pathology results for doctors.
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

    final_response = llm_call(system_prompt_2, "Generate a concise response to the doctor's query based on the analysis results.")

    return final_response

def llm_call(system_prompt, user_prompt):
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        # "model": "llama-3.2-1b-instruct",
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
    lines = response.split('\n')
    parsed = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            parsed[key.strip()] = value.strip()
    return parsed

def process_general_pathology(image_data, ai_assistant_mode=False, requested_entities=None):
    model = GeneralPathologyModel()
    _, logs = model.process_image(image_data, ai_assistant_mode=ai_assistant_mode, requested_entities=requested_entities)
    return logs

def process_retinal_vessel_segmentation(image_data):
    model = RetinalVesselSegmentation()
    segmented_image, logs = model.process_image(image_data)
    return segmented_image, logs
    
def process_od_oc_segmentation(image_data):
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
    match = re.search(f"{pathology}: (0\.\d+)", result, re.IGNORECASE)
    return match.group(1) if match else "N/A"

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def load_users():
    users = {}
    if os.path.exists(USER_DB):
        with open(USER_DB, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                users[row[0]] = row[1]
    return users

def save_user(email, password):
    with open(USER_DB, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([email, password])
        
def save_conversation(doctor_email, patient_id, query, response):
    conversation_data = {
        'conversation_id': str(uuid.uuid4()),
        'doctor_email': doctor_email,
        'patient_id': patient_id,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'query': query,
        'response': response
    }
    
    df = pd.DataFrame([conversation_data])
    df.to_csv(CONVERSATION_DB, mode='a', header=not os.path.exists(CONVERSATION_DB), index=False)
    
def register_user(email, password):
    users = load_users()
    if email in users:
        return False
    if len(password) < 5:
        return False
    save_user(email, password)
    return True

def login_user(email, password):
    users = load_users()
    return email in users and users[email] == password

def resize_image(image, max_size=(300, 300)):
    """Resize image while maintaining aspect ratio"""
    image.thumbnail(max_size)
    return image

def load_patients():
    if os.path.exists(PATIENT_DB):
        return pd.read_csv(PATIENT_DB)
    return pd.DataFrame(columns=['patient_id', 'name', 'age', 'gender', 'weight', 'blood_group', 'bmi', 'height_cm', 'doctor_email'])

def save_patient(patient_data):
    df = load_patients()
    if 'patient_id' not in patient_data:
        patient_data['patient_id'] = str(uuid.uuid4())
    # Add doctor_email to patient data
    patient_data['doctor_email'] = st.session_state.user_email
    df = pd.concat([df, pd.DataFrame([patient_data])], ignore_index=True)
    df.to_csv(PATIENT_DB, index=False)

def add_patient_info():
    st.subheader("New Patient Information")
    with st.form("patient_form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=0, max_value=120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, step=0.1, value=75.0)
        blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        height_cm = st.number_input("Height (cm)", min_value=0.0, max_value=300.0, step=0.1, value=170.0)
        
        bmi = weight / ((height_cm/100) ** 2) if height_cm > 0 else 0
        st.write(f"Calculated BMI: {bmi:.2f}")

        if st.form_submit_button("Save Patient Information"):
            patient_data = {
                'name': name,
                'age': age,
                'gender': gender,
                'weight': weight,
                'blood_group': blood_group,
                'height_cm': round(height_cm, 1),
                'bmi': round(bmi, 2)
            }
            save_patient(patient_data)
            st.success("Patient information saved successfully!")

def delete_patient(patient_id):
    patients = load_patients()
    diagnoses = load_diagnoses()
    
    # Remove patient from patients DataFrame
    patients = patients[patients['patient_id'] != patient_id]
    patients.to_csv(PATIENT_DB, index=False)
    
    # Remove patient's diagnoses
    diagnoses = diagnoses[diagnoses['patient_id'] != patient_id]
    diagnoses.to_csv(DIAGNOSIS_DB, index=False)
    
    # Remove patient's images
    if os.path.exists('images'):
        for file in os.listdir('images'):
            if file.startswith(f"original_{patient_id}_") or file.startswith(f"processed_{patient_id}_"):
                os.remove(os.path.join('images', file))

def view_patients():
    st.subheader("Existing Patients Information")
    patients = load_patients()
    # Filter patients by logged-in doctor's email
    patients = patients[patients['doctor_email'] == st.session_state.user_email]
    diagnoses = load_diagnoses()

    if patients.empty:
        st.warning("No patients found.")
    else:
        for _, patient in patients.iterrows():
            with st.expander(f"Patient: {patient['name']}"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Remove doctor_email from display if you don't want it shown
                    display_data = patient.drop('doctor_email').to_frame().T
                    st.write(display_data)
                with col2:
                    if st.button("Delete Patient", key=f"delete_{patient['patient_id']}"):
                        delete_patient(patient['patient_id'])
                        st.success(f"Patient {patient['name']} deleted successfully!")
                        st.rerun()
                
                patient_diagnoses = diagnoses[diagnoses['patient_id'] == patient['patient_id']]
                if not patient_diagnoses.empty:
                    st.subheader("Diagnoses")
                    for index, diagnosis in patient_diagnoses.iterrows():
                        st.write(f"Diagnosis Date: {diagnosis['diagnosis_date']}")
                        st.write(f"Diagnosis Type: {diagnosis['diagnosis_type']}")
                        st.write(f"Doctor: {diagnosis['doctor_email']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Original Image (Click to zoom)")
                            if pd.notna(diagnosis['image_path']):
                                try:
                                    original_image = load_image(diagnosis['image_path'])
                                    image_zoom(original_image)
                                except FileNotFoundError:
                                    st.info("Original image not found.")
                            else:
                                st.info("No original image for this diagnosis.")
                        with col2:
                            if diagnosis['diagnosis_type'] == 'General Pathology':
                                st.write("Processed Image")
                                st.info("No processed image for General Pathology diagnosis.")
                            else:
                                st.write("Processed Image (Click to zoom)")
                                if pd.notna(diagnosis['processed_image_path']):
                                    try:
                                        processed_image = load_image(diagnosis['processed_image_path'])
                                        image_zoom(processed_image)
                                    except FileNotFoundError:
                                        st.info("Processed image not found.")
                                else:
                                    st.info("No processed image for this diagnosis.")
                        st.subheader("Model Logs")
                        st.text(diagnosis['logs'])
                        st.markdown("---")
                else:
                    st.info("No diagnoses found for this patient.")
                    
def load_diagnoses():
    if os.path.exists(DIAGNOSIS_DB):
        return pd.read_csv(DIAGNOSIS_DB)
    return pd.DataFrame(columns=['diagnosis_id', 'patient_id', 'doctor_email', 'image_path', 'processed_image_path', 'logs', 'diagnosis_date', 'diagnosis_type'])

def save_diagnosis(diagnosis_data):
    df = load_diagnoses()
    if 'diagnosis_id' not in diagnosis_data:
        diagnosis_data['diagnosis_id'] = str(uuid.uuid4())
    df = pd.concat([df, pd.DataFrame([diagnosis_data])], ignore_index=True)
    df.to_csv(DIAGNOSIS_DB, index=False)

@cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)
    return processor, model, device

def transcribe_audio(audio):
    processor, model, device = load_whisper_model()
    
    ensure_recordings_folder()
    
    # Create a temporary file in the recordings folder
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="recordings") as temp_audio:
        temp_audio_path = temp_audio.name
        audio.export(temp_audio_path, format="wav")
    
    try:
        # Load the audio file and resample to 16000 Hz
        audio_array, original_sr = librosa.load(temp_audio_path, sr=None)
        audio_array = librosa.resample(audio_array, orig_sr=original_sr, target_sr=16000)
        
        # Ensure audio is in float32 and scaled to [-1, 1]
        audio_array = audio_array.astype(np.float32)
        if audio_array.max() > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Process the audio
        input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        
        # Generate token ids
        predicted_ids = model.generate(input_features)
        
        # Decode the token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return transcription[0]
    
    finally:
        # Clean up the temporary file
        os.unlink(temp_audio_path)

def save_current_diagnosis(selected_patient_id):
    # Save original image
    original_image_path = save_image(st.session_state.uploaded_image, selected_patient_id, 'original')
    processed_image_path = None

    # Save processed image if it exists
    if hasattr(st.session_state, 'processed_image') and st.session_state.processed_image is not None:
        # Convert to PIL Image if it's a numpy array
        if isinstance(st.session_state.processed_image, np.ndarray):
            processed_pil_image = PILImage.fromarray(st.session_state.processed_image.astype('uint8'))
        else:
            processed_pil_image = st.session_state.processed_image
        processed_image_path = save_image(processed_pil_image, selected_patient_id, 'processed')

    # Create diagnosis data from current state
    diagnosis_data = {
        'patient_id': selected_patient_id,
        'doctor_email': st.session_state.user_email,
        'image_path': original_image_path,
        'processed_image_path': processed_image_path,
        'logs': st.session_state.model_logs if hasattr(st.session_state, 'model_logs') else "",
        'diagnosis_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'diagnosis_type': st.session_state.current_diagnosis_type
    }

    # Save to database
    save_diagnosis(diagnosis_data)

def image_upload_and_diagnosis():
    st.subheader("Image Upload and Diagnosis")
    patients = load_patients()
    patients = patients[patients['doctor_email'] == st.session_state.user_email]
    patient_names = patients['name'].tolist()
    patient_ids = patients['patient_id'].tolist()
    
    if not patient_names:
        st.warning("No patients found. Please add a patient before proceeding with diagnosis.")
        return

    selected_patient = st.selectbox("Select Patient", patient_names)
    selected_patient_id = patient_ids[patient_names.index(selected_patient)]
    
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
        st.session_state.display_image = None
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        original_image = PILImage.open(uploaded_file)
        display_image = original_image.copy()
        display_image.thumbnail((512, 512))
        st.session_state.uploaded_image = original_image
        st.session_state.display_image = display_image
    
    if st.session_state.uploaded_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.write("Click on the image to zoom")
            image_zoom(st.session_state.display_image)
        
        with col2:
            tab1, tab2 = st.tabs(["ChákṣuAI Assistant", "Detailed Diagnosis"])
            
            with tab1:
                st.subheader("ChákṣuAI Assistant")
                input_method = st.radio("Choose input method:", ["Text", "Voice"])
                
                if 'current_llm_response' not in st.session_state:
                    st.session_state.current_llm_response = None
                
                if input_method == "Text":
                    user_input = st.text_input("Ask a question about the diagnosis:")
                    if st.button("Submit"):
                        with st.spinner("Generating response..."):
                            image_data = BytesIO()
                            st.session_state.uploaded_image.save(image_data, format='PNG')
                            image_data = image_data.getvalue()
                            st.session_state.current_llm_response = call_llm_api(user_input, image_data)
                            save_conversation(st.session_state.user_email, selected_patient_id, user_input, st.session_state.current_llm_response)
                            st.session_state.current_diagnosis_type = "ChákṣuAI Assistant"
                            st.session_state.processing_done = True
                
                else:  # Voice mode
                    st.write("Click 'Start Recording' and speak your question:")
                    audio = audiorecorder("Start Recording", "Stop Recording")
                    
                    # Initialize audio_changed in session state if not present
                    if 'previous_audio_length' not in st.session_state:
                        st.session_state.previous_audio_length = 0
                    
                    # Check if we have new audio
                    current_audio_length = len(audio)
                    audio_changed = current_audio_length != st.session_state.previous_audio_length
                    st.session_state.previous_audio_length = current_audio_length
                    
                    if current_audio_length > 0 and audio_changed:
                        with st.spinner("Processing your question..."):
                            user_input = transcribe_audio(audio)
                            st.write(f"Transcribed question: {user_input}")
                            
                            image_data = BytesIO()
                            st.session_state.uploaded_image.save(image_data, format='PNG')
                            image_data = image_data.getvalue()
                            
                            st.session_state.current_llm_response = call_llm_api(user_input, image_data)
                            save_conversation(st.session_state.user_email, selected_patient_id, user_input, st.session_state.current_llm_response)
                            st.session_state.current_diagnosis_type = "ChákṣuAI Assistant"
                            st.session_state.processing_done = True

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
                
    else:
        st.info("Please upload an image to proceed with diagnosis.")


def create_pdf_report(patient_data, diagnoses, doctor_email):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    elements.append(Paragraph("CháksuAI Diagnostic Report", title_style))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Patient Information", subtitle_style))
    height_cm = patient_data['height_cm']
    height_ft, height_in = divmod(height_cm / 2.54, 12)
    
    patient_info = [
        ["Name", patient_data['name']],
        ["Age", str(patient_data['age'])],
        ["Gender", patient_data['gender']],
        ["Weight", f"{patient_data['weight']} kg"],
        ["Height", f"{patient_data['height_cm']:.1f} cm"],
        ["BMI", f"{patient_data['bmi']:.2f}"],
        ["Blood Group", patient_data['blood_group']]
    ]

    patient_table = Table(patient_info)
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Diagnosis", subtitle_style))
    check_one = 1
    for _, diagnosis in diagnoses.iterrows():

        
        if pd.notna(diagnosis['image_path']) and check_one:
            check_one = 0
            elements.append(Paragraph(f"Diagnosis Date: {diagnosis['diagnosis_date']}", normal_style))
            elements.append(Paragraph(f"Doctor: {diagnosis['doctor_email']}", normal_style))
            elements.append(Paragraph("Original Image:", normal_style))
            elements.append(Spacer(1, 6))
            image = load_image(diagnosis['image_path'])
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            elements.append(ReportLabImage(img_buffer, width=200, height=200))
        
        if pd.notna(diagnosis['processed_image_path']) and diagnosis['diagnosis_type'] != 'General Pathology':
            elements.append(Paragraph(f"Diagnosis Type: {diagnosis['diagnosis_type']}", normal_style))
            elements.append(Paragraph("Details:", normal_style))
            elements.append(Paragraph(diagnosis['logs'], normal_style))
            elements.append(Paragraph("Processed Image:", normal_style))
            image = load_image(diagnosis['processed_image_path'])
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            elements.append(ReportLabImage(img_buffer, width=200, height=200))

        

        elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("CháksuAI Assistant Conversations", subtitle_style))
    conversations = pd.read_csv(CONVERSATION_DB)
    patient_conversations = conversations[
        (conversations['doctor_email'] == doctor_email) & 
        (conversations['patient_id'] == patient_data['patient_id'])
    ]
    
    if not patient_conversations.empty:
        for _, conv in patient_conversations.iterrows():
            elements.append(Paragraph(f"Timestamp: {conv['timestamp']}", normal_style))
            elements.append(Paragraph(f"Query: {conv['query']}", normal_style))
            elements.append(Paragraph(f"Response: {conv['response']}", normal_style))
            elements.append(Spacer(1, 12))
    else:
        elements.append(Paragraph("No conversations found for this patient.", normal_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_report(doctor_email):
    st.subheader("Generate Diagnostic Report")
    
    patients = load_patients()
    # Filter patients by logged-in doctor's email
    patients = patients[patients['doctor_email'] == doctor_email]
    patient_names = patients['name'].tolist()
    patient_ids = patients['patient_id'].tolist()
    
    if not patient_names:
        st.warning("No patients found.")
        return
        
    selected_patient = st.selectbox("Select Patient", patient_names)
    selected_patient_id = patient_ids[patient_names.index(selected_patient)]
    
    if st.button("Generate PDF Report"):
        patient_data = patients[patients['patient_id'] == selected_patient_id].iloc[0]
        diagnoses = load_diagnoses()
        patient_diagnoses = diagnoses[diagnoses['patient_id'] == selected_patient_id]
        
        if patient_diagnoses.empty:
            st.warning("No diagnosis found for this patient.")
        else:
            pdf_buffer = create_pdf_report(patient_data, patient_diagnoses, doctor_email)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"{selected_patient}_report.pdf",
                mime="application/pdf"
            )
            st.success("PDF report generated successfully!")
            
def main_app():
    with st.sidebar:
        st.title(f"Welcome, Dr. {st.session_state.user_email}")
        page = st.radio("Dashboard", ["Add New Patient", "View Existing Patients", "Image Upload and Diagnosis", "Generate Diagnostic Report"])
        if st.button("Logout", key="logout"):
            st.session_state.logged_in = False
            st.rerun()
    
    if page == "Add New Patient":
        add_patient_info()
    elif page == "View Existing Patients":
        view_patients()
    elif page == "Image Upload and Diagnosis":
        image_upload_and_diagnosis()
    elif page == "Generate Diagnostic Report":
        generate_report(st.session_state.user_email)

def home_page():
    # Header with navigation
    col1, col2, col3, col4, col5, col6 = st.columns([5, 1, 2, 2, 1, 1])
    with col1:
        st.markdown("<h1 style='font-size: 24px; color: #1E88E5;'>ChákṣuAI: An AI Assistant for Comprehensive Ophthalmic Diagnostics</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("<a href='#about' class='nav-button'>About</a>", unsafe_allow_html=True)
    with col3:
        st.markdown("<a href='#papers' class='nav-button'>Why ChákṣuAI?</a>", unsafe_allow_html=True)
    with col4:
        st.markdown("<a href='#work' class='nav-button'>Our Prior Work</a>", unsafe_allow_html=True)
    with col5:
        if st.button("Login"):
            st.session_state.page = "login"
            st.rerun()
    with col6:
        if st.button("Register"):
            st.session_state.page = "register"
            st.rerun()

    # Navigation buttons
    st.markdown("""
        <style>
        .nav-button {
            display: inline-block;
            padding: 10px 20px;
            margin: 10px;
            text-decoration: none;
            color: #1E88E5;
            cursor: pointer;
        }
        .section-header {
            color: #1E88E5;
            font-size: 24px;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            color: #666;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)


    # About Section
    st.markdown("<h2 class='section-header' id='about'>About</h2>", unsafe_allow_html=True)
    st.write("""
    The ChákṣuAI project aims to develop an open-source, AI-powered assistant to support ophthalmologists in
    diagnosing and managing various eye diseases, including diabetic retinopathy, glaucoma, and age-related macular degeneration 
    (AMD). Using state-of-the-art deep learning models, ChákṣuAI will provide functionalities including classification of retinal 
    diseases (e.g., diabetic retinopathy, glaucoma), detection of abnormalities (e.g., microaneurysms, haemorrhages), retinal 
    segmentation (e.g., optic disc and cup), volumetric analysis of fluid in optical coherence tomography (OCT) images, and risk 
    prediction for disease progression. These capabilities will be integrated into a unified platform to enhance diagnostic accuracy 
    and efficiency, particularly in low-resource settings.    Chaksu AI is a cutting-edge platform that leverages artificial 
    intelligence for automated pathology and segmentation of retinal fundus images. Our system assists medical professionals in quick and 
    accurate diagnosis of various retinal conditions, improving healthcare outcomes for patients worldwide.
    """)

    # Why ChákṣuAI 
    st.markdown("<h2 class='section-header' id='work'>Why ChákṣuAI?</h2>", unsafe_allow_html=True)
    st.write("""
    According to the most recent National Blindness and Visual Impairment Survey (2015-2019), an estimated 4.8 million people in India 
    are blind, while another 29.2 million suffer from moderate or severe visual impairment. The socio-economic burden of poor eye health 
    is significant, imposing an estimated cost of 1.158 lakh crore rupees in 2019, representing about 0.57% of India’s GDP. This 
    burden affects not only economic productivity but also quality of life, including increased caregiver demands, educational loss
    , and reduced productivity in both paid and unpaid work. ChákṣuAI aims to address these challenges by providing a cost-effective, 
    accessible diagnostic solution that reduces the incidence of preventable blindness and mitigates the broader socio-economic impacts of 
    visual impairment. By automating routine diagnostic tasks, ChákṣuAI will enable ophthalmologists to focus on more complex cases, thus 
    increasing patient throughput and reducing diagnostic backlogs. The open-source nature of the project ensures accessibility for smaller 
    clinics and resource-limited areas, ultimately contributing to improved quality of life for millions of individuals. Additionally, the 
    AI assistant will serve as a valuable educational tool for young ophthalmologists, enhancing their diagnostic skills and familiarity 
    with AI-driven healthcare technologies, thus also building a stronger foundation for the future of eye care in India.

    """)

    # Papers Section
    st.markdown("<h2 class='section-header' id='papers'>Our Prior Work</h2>", unsafe_allow_html=True)
    st.write("""
    1. Harishwar Reddy Kasireddy, Udaykanth Reddy Kallam, Sowmitri Karthikeya Siddhartha Mantrala,
    Hemanth Kongara, Anshul Shivhare, Jayesh Saita, Sharanya Vijay, Raghu Prasad, Rajiv Raman,
    and Chandra Sekhar Seelamantula. Deep-learning-based visualization and volumetric analysis of fluid
    regions in optical coherence tomography scans. Diagnostics, 13(16):2659, 2023.
    2. JR Harish Kumar, Chandra Sekhar Seelamantula, Yogish Subraya Kamath, and Rajani Jampala. Rim-
    to-disc ratio outperforms cup-to-disc ratio for glaucoma prescreening. Scientific reports, 9(1):7099, 2019.
    3. JR Harish Kumar, Chandra Sekhar Seelamantula, Ashwin Mohan, Rohit Shetty, TJM Berendschot, and
    Carroll AB Webers. Automatic analysis of normative retinal oximetry images. Plos one, 15(5):e0231677, 2020.
    4. Aniketh Manjunath, Subramanya Jois, and Chandra Sekhar Seelamantula. Robust segmentation of
    optic disc and cup from fundus images using deep neural networks. arXiv preprint arXiv:2012.07128, 2020.
    5. Dhruv Mohan, JR Harish Kumar, and Chandra Sekhar Seelamantula. Optic disc segmentation using
    cascaded multiresolution convolutional neural networks. In 2019 IEEE International Conference on
    Image Processing (ICIP), pages 834–838. IEEE, 2019.
    6. P Kevin Raj, JR Harish Kumar, Subramanya Jois, S Harsha, and Chandra Sekhar Seelamantula. A
    structure tensor based voronoi decomposition technique for optic cup segmentation. In 2019 IEEE
    International Conference on Image Processing (ICIP), pages 829–833. IEEE, 2019.
    7. P Kevin Raj, Aniketh Manjunath, JR Harish Kumar, and Chandra Sekhar Seelamantula. Automatic
    classification of artery/vein from single wavelength fundus images. In 2020 IEEE 17th International
    Symposium on Biomedical Imaging (ISBI), pages 1262–1265. IEEE, 2020.
    8. JR Harish Kumar, Chandra Sekhar Seelamantula, JH Gagan, Yogish S Kamath, Neetha IR Kuzhuppilly,
    U Vivekanand, Preeti Gupta, and Shilpa Patil. Ch´aks.u: A glaucoma specific fundus image database.
    Scientific data, 10(1):70, 2023.
    """)

    # Footer
    st.markdown("""
        <div class='footer'>
            Developed by Spectrum Lab @ IISc Bangalore
        </div>
    """, unsafe_allow_html=True)

def login_page():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("Login")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
            
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if login_user(email, password):
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

def register_page():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("Register")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
            
        with st.form("register_form"):
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            submit_button = st.form_submit_button("Register")
            
            if submit_button:
                if not is_valid_email(email):
                    st.error("Invalid email address")
                elif len(password) < 5:
                    st.error("Password must be at least 5 characters long")
                elif register_user(email, password):
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.success("Registered and logged in successfully!")
                    st.rerun()
                else:
                    st.error("Email already registered")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.logged_in:
        main_app()
    else:
        if st.session_state.page == "home":
            home_page()
        elif st.session_state.page == "login":
            login_page()
        elif st.session_state.page == "register":
            register_page()

if __name__ == "__main__":
    main()