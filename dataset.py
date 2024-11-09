import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torchvision import transforms

class COCODataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None, max_caption_length=50):
        self.image_dir = image_dir
        self.transform = transform
        self.max_caption_length = max_caption_length

        # Load captions
        with open(caption_file, 'r') as f:
            self.captions_data = json.load(f)['annotations']
        
        self.image_ids = [item['image_id'] for item in self.captions_data]
        self.captions = [item['caption'] for item in self.captions_data]

        # Define a transform if not provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f'{image_id:012}.jpg')

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert caption to tokens (here assume some tokenization)
        caption_tokens = [2] + [ord(char) for char in caption[:self.max_caption_length - 2]] + [3]  # Add start and end tokens

        # Pad caption tokens to max_caption_length
        if len(caption_tokens) < self.max_caption_length:
            caption_tokens += [0] * (self.max_caption_length - len(caption_tokens))

        caption_tensor = torch.tensor(caption_tokens)
        return image, caption_tensor

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions = torch.stack(captions)
    return images, captions
