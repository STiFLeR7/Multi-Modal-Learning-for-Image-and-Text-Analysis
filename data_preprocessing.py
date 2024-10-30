import json
import os
from PIL import Image
import torch
from torchvision import transforms

class DataPreprocessor:
    def __init__(self, image_dir, caption_file):
        self.image_dir = image_dir
        self.caption_file = caption_file
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        with open(self.caption_file, 'r') as f:
            captions_data = json.load(f)
        
        captions = {}
        for ann in captions_data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            if image_id in captions:
                captions[image_id].append(caption)
            else:
                captions[image_id] = [caption]
                
        return captions

    def process_image(self, image_id):
        image_path = os.path.join(self.image_dir, f"{image_id:012d}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)
