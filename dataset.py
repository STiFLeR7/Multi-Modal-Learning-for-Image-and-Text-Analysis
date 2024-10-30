import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class COCODataset(Dataset):
    def __init__(self, image_dir, captions, transform=None):
        self.image_dir = image_dir
        self.captions = captions
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_id = list(self.captions.keys())[idx]
        caption = self.captions[image_id][0]  # Pick the first caption

        image_path = os.path.join(self.image_dir, f"{image_id:012d}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Dummy caption tensor for demonstration, replace with tokenization as needed
        caption_tensor = torch.randint(0, 30522, (32,))  # Assuming BERT-like vocab size and max length
        
        return image, caption_tensor
