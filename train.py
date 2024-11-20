import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from vit_model import VisionTransformer  # Assuming this is correct import
from coco_loader import get_coco_dataloader

# Define paths
root_dir = 'D:/COCO-DATASET/coco2017/train2017'
annotation_file = 'D:/COCO-DATASET/coco2017/annotations/captions_train2017.json'

# Transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model and criterion
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(pretrained=True, num_classes=80).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Tokenizer definition
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"

    def encode(self, caption):
        # Split caption into words and convert to indices
        return [self.word_to_idx.get(word, self.word_to_idx["<UNK>"]) for word in caption.split()]

    def decode(self, indices):
        # Convert indices back to words
        return ' '.join([self.idx_to_word[idx] for idx in indices])

# Example vocab, you should load it from your dataset
vocab = ["<PAD>", "<START>", "<END>", "<UNK>"] + ["word1", "word2", "word3", "word4"]  # Extend this with actual vocabulary
tokenizer = Tokenizer(vocab)

# Define max caption length for padding and truncation
max_caption_length = 11  # Adjusted to the expected length

def pad_sequence(captions, max_length):
    # Pad or truncate each caption to max_length
    padded_captions = []
    for caption in captions:
        # Truncate the caption if it exceeds max_length
        caption = caption[:max_length]
        # Pad the caption if it's shorter than max_length
        padded_caption = caption + [tokenizer.word_to_idx["<PAD>"]] * (max_length - len(caption))
        padded_captions.append(padded_caption)
    return padded_captions

def train_model():
    # Load the dataloaders
    train_loader, val_loader, _ = get_coco_dataloader(
        batch_size=32, transform=transform, root_dir=root_dir, annotation_file=annotation_file)

    # Training loop with print statements for debugging
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(DEVICE)

            # captions is a list of tuples (image_id, caption)
            # Extract just the captions from the list of tuples
            captions = [caption[1] for caption in captions]  # caption[1] contains the actual text

            # Tokenize captions: convert string captions to integer token indices
            captions = [tokenizer.encode(caption) for caption in captions]

            # Pad or truncate captions to a fixed length
            captions = pad_sequence(captions, max_caption_length)

            # Check if batch size of images and captions match
            print(f"Batch [{batch_idx}], Image batch size: {images.size(0)}, Caption batch size: {len(captions)}")
            assert images.size(0) == len(captions), "Batch size of images and captions do not match."

            # Convert captions to tensor and move to device
            captions = torch.tensor(captions).to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Assume the model is a classifier and captions are the targets (for classification)
            # CrossEntropyLoss expects target labels to be class indices
            loss = criterion(outputs, captions)  # Captions should be class indices (integer labels)

            # Backward pass
            loss.backward()

            # Optimize the model
            optimizer.step()

            running_loss += loss.item()

            # Debugging print statement
            if batch_idx % 10 == 0:  # Print every 10th batch
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {running_loss/(batch_idx+1):.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average loss: {running_loss/len(train_loader):.4f}")

if __name__ == '__main__':
    train_model()
