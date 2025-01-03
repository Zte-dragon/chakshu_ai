from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from io import BytesIO
from flair import FLAIRModel
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
import base64
GeneralPathologyModel_path = '/home/sasi/puneet/chakshu_ai_2/model_weights/flair_resnet.pth'
ratinasegmentationmodel_path = "/home/sasi/puneet/chakshu_ai_2/model_weights/best.pt"
ratina_vessel_state_dict = "/home/sasi/puneet/chakshu_ai_2/model_weights/ratina_vessel_state_dict.pth"
class BaseModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def process_image(self, image_data, is_numpy=False):
        pass
    
    def preprocess_image_data(self, image_data):
        if isinstance(image_data, np.ndarray):
            # Already a numpy array
            return image_data
        elif isinstance(image_data, str):  # base64
            # Convert base64 to numpy
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            return np.array(image)
        elif isinstance(image_data, bytes):  # raw binary
            # Convert binary to numpy
            image = Image.open(BytesIO(image_data)).convert('RGB')
            return np.array(image)
        else:
            raise ValueError("Unsupported image format")
        

class RetinaSegmentationModel(BaseModel):
    def __init__(self):
        super().__init__("Retina Segmentation")
        self.model = YOLO(ratinasegmentationmodel_path)

    def flatten_points(self, points):
        return [(point[0][0], point[0][1]) for point in points]

    def calculate_distances(self, tuple1, tuple2):
        distances = []
        
        for point1 in tuple1:
            for point2 in tuple2:
                distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                distances.append(distance)

        min_distance = min(distances)
        max_distance = max(distances)
        
        return min_distance, max_distance

    def process_image(self, image_data, is_numpy=False):
        # if not is_numpy:
        #     # Handle BytesIO input for backward compatibility
        #     image = Image.open(BytesIO(image_data)).convert('RGB')  # Convert to RGB
        #     image_np = np.array(image)
        # else:
        #     # Direct numpy array input
        #     image = Image.fromarray(image_data).convert('RGB')  # Convert to RGB
        #     image_np = np.array(image)

        image_np = self.preprocess_image_data(image_data)

        results = self.model.predict(image_np, retina_masks=True, show_boxes=False)

        logs = []
        processed_image = image_np.copy()
        
        if len(results) == 0:
            logs.append("No optic structures detected in the image")
            return processed_image, "\n".join(logs)
            
        for r in results:
            try:
                flat_disc_contour = r.masks.xy[0]
                disc_contour = np.array(flat_disc_contour).reshape((-1, 1, 2)).astype(np.int32)
                flat_cup_contour = r.masks.xy[1]
                cup_contour = np.array(flat_cup_contour).reshape((-1, 1, 2)).astype(np.int32)
                
                _, disc_max = self.calculate_distances(flat_disc_contour, flat_disc_contour)
                _, cup_max = self.calculate_distances(flat_cup_contour, flat_cup_contour)
                disc_cup_min, _ = self.calculate_distances(flat_disc_contour, flat_cup_contour)

                logs.append(f"Disc max: {disc_max:.2f}, Cup max: {cup_max:.2f}, Rim min: {disc_cup_min:.2f}")
                logs.append(f"CDR: {cup_max/disc_max:.2f}, RDR: {disc_cup_min/disc_max:.2f}")
                
                processed_image = cv2.drawContours(processed_image, [disc_contour, cup_contour], -1, (0, 0, 255), 10)
            except IndexError as e:
                logs.append(f"IndexError: {str(e)}")
        print(f"**********\n{logs}\n**********")
        return processed_image, "\n".join(logs)

class RetinalVesselSegmentation(BaseModel):
    def __init__(self):
        super().__init__("Retinal Vessel Segmentation")
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
        self.model.load_state_dict(torch.load(ratina_vessel_state_dict))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_data, is_numpy=False):
        # if not is_numpy:
        #     # Handle BytesIO input for backward compatibility
        #     image = Image.open(BytesIO(image_data)).convert('RGB')  # Convert to RGB
        #     image_np = np.array(image)
        # else:
        #     # Direct numpy array input
        #     image = Image.fromarray(image_data).convert('RGB')  # Convert to RGB
        #     image_np = np.array(image)

        image_np = self.preprocess_image_data(image_data)
        
        # Store original size
        original_size = image_np.shape[:2][::-1]  # Convert (H,W) to (W,H)
        
        # Preprocess the image
        input_image = self.transform(Image.fromarray(image_np)).unsqueeze(0)  # Add batch dimension

        # Perform segmentation
        with torch.no_grad():
            output = self.model(input_image)
            output = torch.sigmoid(output).squeeze().cpu().numpy()

        # Resize output back to original size if needed
        if output.shape != (original_size[1], original_size[0]):
            output = cv2.resize(output, (original_size[0], original_size[1]))

        # Threshold the output to create a binary mask
        threshold = 0.5
        binary_mask = output > threshold

        # Create a blue mask where vessels are detected
        segmented_image = image_np.copy()  # Use the already converted numpy array
        segmented_image[binary_mask] = [0,0,255]
        segmented_image = segmented_image.astype(np.uint8)

        # Calculate vessel density
        vessel_density = np.mean(binary_mask) * 100

        logs = f"Vessel Density: {vessel_density:.2f}%"

        return segmented_image, logs

class GeneralPathologyModel(BaseModel):
    def __init__(self):
        super().__init__("General Pathology")
        self.model = FLAIRModel(from_checkpoint=True, weights_path=GeneralPathologyModel_path)
        self.text_categories = [
            "Normal","Age-Related Macular Degeneration", "Macular Edema", "Diabetic Retinopathy",
            "Glaucoma","Cataract","Retinal Vein Occlusion","Lesion in the Macula", 'Retinal Detachment','Hypertensive Retinopathy'
        ]

    def process_image(self, image_data, is_numpy=False, ai_assistant_mode=False, requested_entities=None):
        # if not is_numpy:
        #     # Handle BytesIO input for backward compatibility
        #     image = Image.open(BytesIO(image_data)).convert('RGB')  # Convert to RGB
        #     image_np = np.array(image)
        # else:
        #     # Direct numpy array input
        #     image = Image.fromarray(image_data).convert('RGB')  # Convert to RGB
        #     image_np = np.array(image)

        image_np = self.preprocess_image_data(image_data)

        if ai_assistant_mode:
            if requested_entities == 'ALL':
                categories = self.text_categories
            else:
                categories = ['normal']
                for entity in requested_entities:
                    categories.extend([f"{entity.strip()}"])
        else:
            categories = self.text_categories

        print(f"categories being used: \n {categories}")
        # Forward FLAIR model to compute similarities
        probs, logits = self.model(image_np, categories)

        # Convert probs to a 1D numpy array
        probs = probs.squeeze()
        # Generate logs
        logs = []
        for category, prob in zip(categories, probs):
            logs.append(f"{category}: {100*prob:.2f}%")
        print(logs)
        return None, "\n".join(logs)
