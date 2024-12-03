import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image

# Set paths
MODEL_PATH = './coco_resnet18.pth'
COCO_DATASET_DIR = './preprocessed_data/'  # Adjust if needed

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
def load_model(model_path):
    model = resnet18(num_classes=80)  # Adjust the number of classes if needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocessing pipeline for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load .npy images
def load_image_from_npy(npy_path):
    image_array = np.load(npy_path)
    
    # Handle different shapes of image data
    if image_array.ndim == 4:
        # Assuming the shape is (1, 224, 224, 3), remove the batch dimension
        image_array = image_array[0]
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        # Image already has the correct shape (224, 224, 3)
        pass
    else:
        raise ValueError(f"Invalid image shape: {image_array.shape}. Expected (224, 224, 3) or (1, 224, 224, 3).")

    # Convert the image to uint8 if it's in float32 (e.g., normalized [0, 1])
    if image_array.dtype == np.float32:
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(image_array)
    return image

# Perform inference
def inference(model, image_path):
    image = load_image_from_npy(image_path)  # Load .npy image
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[predicted_class].item()

# Run inference on COCO dataset
def run_inference_on_dataset(dataset_dir, model):
    # Get all .npy files for images
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(dataset_dir, image_file)
        predicted_class, confidence = inference(model, image_path)

        print(f"Image {idx+1}: Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")

# Main script
if __name__ == '__main__':
    model = load_model(MODEL_PATH)

    # Option 2: Inference on the full dataset
    run_inference_on_dataset(COCO_DATASET_DIR, model)
