import os
import json
import torch
from torch.utils.data import Dataset, DataLoader  # Import DataLoader here
from PIL import Image
from collections import defaultdict

class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.transform = transform
        
        # Load the annotations from the correct file
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create a mapping of image_id to captions
        self.image_id_to_caption = defaultdict(list)
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            self.image_id_to_caption[image_id].append(caption)
        
        self.image_ids = list(self.image_id_to_caption.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get the image id
        image_id = self.image_ids[idx]
        
        # Get the image file path
        img_path = os.path.join(self.root_dir, f'{str(image_id).zfill(12)}.jpg')
        image = Image.open(img_path).convert("RGB")
        
        # Get the caption(s) for this image
        captions = self.image_id_to_caption[image_id]
        
        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)
        
        return image, captions

# Function to load the dataloaders
def get_coco_dataloader(batch_size, transform, root_dir, annotation_file, missing_caption='No caption available'):
    coco_dataset = CocoDataset(root_dir, annotation_file, transform)
    train_size = int(0.8 * len(coco_dataset))
    val_size = len(coco_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(coco_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, {}
