import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.idx = 4

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def __len__(self):
        return len(self.word2idx)

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
        self.vocab = self._build_vocab()
        self.transform = transform

    def _load_captions(self, caption_file):
        """Loads captions from the provided file."""
        captions = []
        with open(caption_file, 'r') as file:
            for line in file:
                image_id, caption = line.strip().split('\t')
                # Remove the '#X' part from the image_id
                image_id = image_id.split('#')[0]
                captions.append((image_id, caption))
        return captions

    def _build_vocab(self):
        """Builds a vocabulary from the captions."""
        vocab = Vocabulary()
        for _, caption in self.captions:
            for word in caption.split():
                vocab.add_word(word)
        return vocab

    def _caption_to_tensor(self, caption):
        """Converts a caption string into a tensor of word indices."""
        tokens = ["<start>"] + caption.split() + ["<end>"]
        return torch.tensor([self.vocab(token) for token in tokens], dtype=torch.long)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_id, caption = self.captions[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption_tensor = self._caption_to_tensor(caption)

        return image, caption_tensor

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
        print("Batch of captions shape:", captions.shape)
        break
