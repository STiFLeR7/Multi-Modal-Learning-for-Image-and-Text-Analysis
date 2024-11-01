import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

class COCODataset(Dataset):
    def __init__(self, image_dir, annotations_file, transform=None, max_caption_length=50):
        self.image_dir = image_dir
        self.max_caption_length = max_caption_length
        
        # Define a default transformation if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to a fixed size
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        self.captions = []
        self.image_ids = []
        
        for ann in annotations['annotations']:
            self.captions.append(ann['caption'])
            self.image_ids.append(ann['image_id'])
        
        # Create a mapping from image_id to file name
        self.image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_id = self.image_ids[idx]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, image_filename)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        
        # Open the image and apply transformations
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Convert caption to a list of indices (a simplified example using ASCII values)
        caption_indices = torch.tensor([ord(c) % 256 for c in caption[:self.max_caption_length]], dtype=torch.long)
        
        return image, caption_indices

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)  # Stack images
    captions = pad_sequence(captions, batch_first=True, padding_value=0)  # Pad captions
    return images, captions
