import os
import tensorflow as tf
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer

# Paths to data directories
IMAGES_PATH = 'D:/Multi-Modal-Learning-for-Image-and-Text-Analysis/coco2017/train2017'  # Update this to your actual path
CAPTIONS_PATH = 'D:/Multi-Modal-Learning-for-Image-and-Text-Analysis/coco2017/annotations/captions_train2017.json'

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Image Preprocessing
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)

# Text Preprocessing
def tokenize_text(caption):
    return tokenizer(caption, padding='max_length', max_length=32, return_tensors="pt").input_ids

# Example Usage: Process all images in the directory
for image_name in os.listdir(IMAGES_PATH):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):  # Adjust for your image formats
        image_path = os.path.join(IMAGES_PATH, image_name)
        sample_image = load_image(image_path)
        print(f"Processed image: {image_name}")
