import torch
from torch.utils.data import DataLoader
from model_architecture import YourModel  # Make sure your model's file is named model_architecture.py
from dataset import COCODataset           # Ensure dataset.py has COCODataset class
import torch.nn as nn

def evaluate(model, dataloader, criterion, device, vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)

            # Get model output (Shape: [batch_size, seq_length, vocab_size])
            outputs = model(images, captions)  # Model outputs shape: (batch_size, seq_length, vocab_size)
            
            # Ensure the batch_size and seq_length match for reshaping
            batch_size, seq_length = captions.size()  # batch_size = 8, seq_length = length of captions
            
            # Check the total number of elements in both the outputs and captions for debugging
            print(f"Outputs shape: {outputs.shape}")  # Debugging output shape
            print(f"Captions shape: {captions.shape}")  # Debugging captions shape
            
            # Flatten the outputs and captions for loss calculation
            try:
                outputs = outputs.view(batch_size * seq_length, vocab_size)  # Flatten: (batch_size * seq_length, vocab_size)
                captions = captions.view(batch_size * seq_length)           # Flatten: (batch_size * seq_length)
            except Exception as e:
                print(f"Error in reshaping: {e}")
                print(f"Number of elements in outputs: {outputs.numel()}")
                print(f"Number of elements in captions: {captions.numel()}")
                continue

            # Compute loss
            loss = criterion(outputs, captions)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Average Evaluation Loss: {avg_loss:.4f}")
    model.train()
    return avg_loss

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 10000  # Update based on your dataset
embed_size = 256
hidden_size = 512
batch_size = 8

# Initialize model and criterion
model = YourModel(vocab_size, embed_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()

# Load your data
image_dir = "D:/COCO-DATASET/coco2017/train2017"
caption_file = "D:/COCO-DATASET/coco2017/annotations/captions_train2017.json"
dataset = COCODataset(image_dir=image_dir, caption_file=caption_file)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Run evaluation
if __name__ == "__main__":
    avg_loss = evaluate(model, dataloader, criterion, device, vocab_size)
    print(f"Evaluation completed. Average Loss: {avg_loss:.4f}")
