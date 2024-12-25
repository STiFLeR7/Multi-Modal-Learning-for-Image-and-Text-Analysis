import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class FlickrDataset(Dataset):
    def __init__(self, image_dir, caption_file, vocab_size, transform=None):
        self.image_dir = image_dir
        self.vocab_size = vocab_size
        self.transform = transform or (lambda x: x)  # Default to identity transform if none provided

        # Load and preprocess captions
        self.data = []
        with open(caption_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    image_name = parts[0]
                    caption = parts[1]
                    self.data.append((image_name, caption))

        print(f"Dataset loaded: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, caption = self.data[idx]

        # Clean image_name: remove everything after '#' and ensure the file ends with '.jpg'
        image_name = image_name.split('#')[0].strip()
        if not image_name.endswith('.jpg'):
            image_name = f"{image_name.split('.')[0]}.jpg"  # Ensure proper extension

        # Construct the image path
        image_path = os.path.join(self.image_dir, image_name)

        # Check if the file exists
        if not os.path.exists(image_path):
            print(f"Warning: File not found -> {image_path}")
            return None, None, None  # Skip missing files gracefully

        # Load and transform the image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None, None

        image = self.transform(image)

        # Tokenize the caption
        caption_tokens = self.convert_caption_to_tokens(caption)
        target = torch.tensor(caption_tokens, dtype=torch.long)  # Ensure target matches seq_len

        return image, torch.tensor(caption_tokens, dtype=torch.long), target

    def convert_caption_to_tokens(self, caption):
        # Example tokenization: Convert each word into a token index
        tokens = caption.split()[:20]  # Truncate to 20 tokens
        token_indices = [random.randint(0, self.vocab_size - 1) for _ in tokens]
        return token_indices + [0] * (20 - len(tokens))  # Pad to seq_len


if __name__ == "__main__":
    # Test the dataset loader
    from torchvision import transforms

    image_dir = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
    caption_file = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"
    vocab_size = 5000

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = FlickrDataset(image_dir, caption_file, vocab_size, transform=transform)
    print(f"Number of samples in dataset: {len(dataset)}")

    # Test a few samples
    for i in range(5):
        image, caption_tokens, target = dataset[i]
        if image is not None:
            print(f"Sample {i}: Image shape: {image.shape}, Caption tokens: {caption_tokens}, Target shape: {target.shape}")
