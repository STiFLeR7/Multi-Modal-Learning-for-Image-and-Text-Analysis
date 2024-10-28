from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
import torchvision.transforms as transforms
import json
import os
import torch

class CustomDataset(Dataset):
    def __init__(self, images_path, captions_path, max_length=20):
        self.images_path = images_path
        self.captions = self.load_captions(captions_path)
        self.image_files = list(self.captions.keys())
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def load_captions(self, captions_path):
        with open(captions_path, 'r') as f:
            annotations = json.load(f)
        captions = {}
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            img_filename = f'{img_id:012d}.jpg'
            captions[img_filename] = ann['caption']
        return captions

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        caption = self.captions[img_name]
        tokens = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens['input_ids'].squeeze(0)  # shape [max_length]

        return image, input_ids  # image shape [3, 224, 224], caption shape [max_length]
