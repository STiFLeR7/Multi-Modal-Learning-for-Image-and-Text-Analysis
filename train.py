import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from vit_model import VisionTransformer  # Assuming this is correct import
from coco_loader import get_coco_dataloader
from collections import defaultdict

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

# Create a simple tokenizer (if you're not using a pretrained tokenizer)
# This is an example, you may want to use a more sophisticated tokenizer for real applications
class SimpleTokenizer:
    def __init__(self):
        self.word_to_idx = defaultdict(lambda: len(self.word_to_idx))  # Automatically assigns new index
        self.word_to_idx["<PAD>"] = 0  # Padding token for sequences
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def encode(self, caption):
        return [self.word_to_idx[word] for word in caption.split()]

    def decode(self, indices):
        return " ".join([self.idx_to_word[idx] for idx in indices])

tokenizer = SimpleTokenizer()

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

            # Tokenize captions: convert string captions to integer token indices
            captions = [tokenizer.encode(caption) for caption in captions]
            captions = torch.tensor(captions).to(DEVICE)  # Convert to tensor and move to device

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
