import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from coco_loader import CocoDataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import json

# Set paths
MODEL_PATH = './coco_resnet18.pth'  # Adjust if needed
COCO_DATASET_DIR = './preprocessed_data/'  # Path to preprocessed data folder
OUTPUT_FILE = './predictions.json'  # File to save the predictions

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

# Save predictions to a JSON file
def save_predictions_to_file(predictions, output_file=OUTPUT_FILE):
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)

# Perform inference
def inference(model, image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure 3-channel input
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[predicted_class].item()

# Run inference on COCO dataset
def run_inference_on_dataset(dataset_dir, model):
    dataset = CocoDataset(
        annotation_file=os.path.join(dataset_dir, 'annotations.json'),  # Ensure this path is correct
        root_dir=os.path.join(dataset_dir, 'images_batch'),  # Path for image data (modify based on batch structure)
        transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    predictions = []
    
    for idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        predictions.append({
            'image_id': idx + 1,  # or use image file name if available
            'predicted_class': predicted_class,
            'confidence': confidence
        })
        
        print(f"Image {idx+1}: Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")
    
    # Save the predictions to file
    save_predictions_to_file(predictions)

# Main script
if __name__ == '__main__':
    # Load the model
    model = load_model(MODEL_PATH)
    
    # Run inference on the dataset
    run_inference_on_dataset(COCO_DATASET_DIR, model)
