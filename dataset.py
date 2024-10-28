import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import BertTokenizer

class CustomDataset(Dataset):
    def __init__(self, images_path, captions_path):
        self.images_path = images_path
        self.captions_path = captions_path
        
        # Load captions data
        with open(captions_path, 'r') as f:
            captions_data = json.load(f)
        
        # Extract image IDs and captions
        self.image_ids = [ann['image_id'] for ann in captions_data['annotations']]
        self.captions = [ann['caption'] for ann in captions_data['annotations']]
        
        # Initialize BERT tokenizer and image transformation
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Load image
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_path, f"{str(image_id).zfill(12)}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Tokenize caption
        caption = self.captions[idx]
        caption_tokens = self.tokenizer(caption, padding='max_length', max_length=32, return_tensors="pt").input_ids
        caption_tokens = caption_tokens.squeeze()  # Remove extra dimension
        
        return image, caption_tokens
