import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from dataset import COCODataset
from model_architecture import MultiModalModel
from data_preprocessing import tokenizer  # Import tokenizer to get vocabulary size

# Define paths for the dataset
image_dir = "D:/COCO-DATASET/coco2017/train2017"
annotation_file = "D:/COCO-DATASET/coco2017/annotations/captions_train2017.json"

# Initialize the dataset and DataLoader
dataset = COCODataset(
    image_dir=image_dir,
    annotation_file=annotation_file,
    transform=None  # Ensures that final_transform from dataset.py is used
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=None)

# Get the vocabulary size from the tokenizer for num_classes
vocab_size = tokenizer.vocab_size

# Initialize the model, loss function, and optimizer
model = MultiModalModel(num_classes=vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move data to device
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        
        # Reshape outputs and targets for CrossEntropyLoss
        outputs = outputs.view(-1, vocab_size)  # Flatten to (batch_size * seq_len, vocab_size)
        targets = input_ids.view(-1)            # Flatten targets to (batch_size * seq_len)

        # Calculate loss and backpropagate
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training complete!")
