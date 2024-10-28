import os
import json
from PIL import Image
import torchvision.transforms as transforms
from transformers import BertTokenizer

IMAGES_PATH = 'D:/Multi-Modal-Learning-for-Image-and-Text-Analysis/coco2017/train2017'  # Update this to your images path
CAPTIONS_PATH = 'D:/Multi-Modal-Learning-for-Image-and-Text-Analysis/coco2017/annotations/captions_train2017.json'  # Update this to your captions path

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and process each image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image)

# Load captions data
with open(CAPTIONS_PATH, 'r') as f:
    captions_data = json.load(f)

# Process captions and save them
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
captions = []
for annotation in captions_data['annotations']:
    caption = annotation['caption']
    tokens = tokenizer(caption, padding='max_length', max_length=32, return_tensors="pt").input_ids
    captions.append(tokens.squeeze().tolist())
    
print("Data preprocessing completed.")
