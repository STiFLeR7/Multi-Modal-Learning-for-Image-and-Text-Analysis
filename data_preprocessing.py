import os
import json
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer

# Define paths
IMAGES_PATH = 'D:/Multi-Modal-Learning-for-Image-and-Text-Analysis/coco2017/train2017'
CAPTIONS_PATH = 'D:/Multi-Modal-Learning-for-Image-and-Text-Analysis/coco2017/annotations/captions_train2017.json'

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer loaded successfully.")

# Image Preprocessing
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)

# Text Preprocessing
def tokenize_text(caption):
    return tokenizer(caption, padding='max_length', max_length=32, return_tensors="pt").input_ids

# Load captions
with open(CAPTIONS_PATH, 'r') as f:
    captions_data = json.load(f)
print("Captions loaded successfully.")

print("Starting data preprocessing...")

# Process images and captions
for idx, annotation in enumerate(captions_data['annotations'][:100]):  # Limiting to 100 samples for demonstration
    image_id = annotation['image_id']
    caption = annotation['caption']
    image_path = os.path.join(IMAGES_PATH, f"{str(image_id).zfill(12)}.jpg")
    if os.path.exists(image_path):
        sample_image = load_image(image_path)
        tokenized_caption = tokenize_text(caption)
        print(f"Processed image {idx + 1}: {image_path}")
    else:
        print(f"Image not found: {image_path}")

print("Data preprocessing completed.")
