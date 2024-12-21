import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class FlickrDataset(Dataset):
    def __init__(self, image_dir, caption_file, vocab_size, transform=None):
        self.image_dir = image_dir
        self.vocab_size = vocab_size
        self.transform = transform or (lambda x: x)  # Use identity function if no transform is provided

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

        # Debugging: Check the image path
        if not os.path.exists(image_path):
            print(f"Debug: Image file does not exist -> {image_path}")

        # Load and transform the image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        # Apply transform if callable
        image = self.transform(image)

        # Convert the caption to token IDs (dummy example)
        caption_tokens = torch.randint(0, self.vocab_size, (20,))  # Dummy tokens
        target = torch.zeros(self.vocab_size)  # Dummy target

        return image, caption_tokens, target

if __name__ == "__main__":
    # Test the dataset loader
    from torchvision import transforms

    image_dir = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
    caption_file = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"
    vocab_size = 5000

    # Define a transform for testing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = FlickrDataset(image_dir, caption_file, vocab_size, transform=transform)
    print(f"Number of samples in dataset: {len(dataset)}")

    # Test a few samples
    for i in range(5):
        try:
            image, caption_tokens, target = dataset[i]
            print(f"Sample {i}: Image size: {image.size()}, Caption tokens: {caption_tokens.shape}")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
