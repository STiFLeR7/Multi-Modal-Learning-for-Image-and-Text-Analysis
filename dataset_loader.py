import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with images.
            caption_file (str): Path to the file with image captions.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.image_dir = image_dir
        self.captions = self._load_captions(caption_file)
        self.transform = transform

    def _load_captions(self, caption_file):
        """Loads captions from the provided file."""
        captions = []
        with open(caption_file, 'r') as file:
            for line in file:
                image_id, caption = line.strip().split('\t')
                captions.append((image_id, caption))
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_id, caption = self.captions[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption

# Define transforms for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset and dataloader
if __name__ == "__main__":
    IMAGE_DIR = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
    CAPTION_FILE = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"

    dataset = Flickr8kDataset(IMAGE_DIR, CAPTION_FILE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Example: Iterate through the dataloader
    for images, captions in dataloader:
        print("Batch of images shape:", images.shape)
        print("Batch of captions:", captions)
        break