import json
import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class CocoDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None):
        self.annotation_file = annotation_file
        self.root_dir = root_dir
        self.transform = transform

        # Load annotations from the JSON file
        with open(self.annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # Prepare lists for images, annotations, and categories
        self.images = self.annotations.get('images', [])
        self.annotations_list = self.annotations.get('annotations', [])
        self.categories = self.annotations.get('categories', [])

        # Create a mapping for category id to category name
        self.category_map = {category['id']: category['name'] for category in self.categories}

        # Filter out invalid annotations
        self.valid_annotations = self._filter_valid_annotations()

    def _filter_valid_annotations(self):
        valid_annotations = []
        for ann in self.annotations_list:
            if 'image_id' in ann:
                image_id = ann['image_id']
                image_info = next((img for img in self.images if img['id'] == image_id), None)
                if image_info:
                    valid_annotations.append(ann)
        return valid_annotations

    def __len__(self):
        return len(self.valid_annotations)

    def __getitem__(self, idx):
        ann = self.valid_annotations[idx]

        image_id = ann['image_id']
        image_info = next(img for img in self.images if img['id'] == image_id)

        # Load the image file
        img_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Get the category id and its name
        category_id = ann['category_id']
        category_name = self.category_map.get(category_id, 'Unknown')

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        return image, category_name  # Return image and category name (or id depending on your needs)
