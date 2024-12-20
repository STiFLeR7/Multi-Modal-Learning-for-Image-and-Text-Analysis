import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import Flickr8kDataset
from model_architecture import CustomModel
from tqdm import tqdm

# Configuration
class Config:
    data_dir = "D:/Flickr8k-Dataset/Flicker8k_Dataset"
    text_file = "D:/Flickr8k-Dataset/Flickr8k_text/Flickr8k.token.txt"
    embedding_dim = 20
    vocab_size = 5000
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001
    model_save_path = "custom_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # Load dataset
    dataset = Flickr8kDataset(Config.data_dir, Config.text_file, Config.vocab_size)
    data_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    # Initialize model, loss, optimizer
    model = CustomModel(Config.embedding_dim, Config.vocab_size).to(Config.device)
    criterion = nn.MSELoss()  # Adjust loss function based on your task
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # Training loop
    model.train()
    for epoch in range(Config.num_epochs):
        epoch_loss = 0
        for images, texts, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}"):
            images, texts, targets = images.to(Config.device), texts.to(Config.device), targets.to(Config.device)
            
            # Forward pass
            outputs = model(images, texts)
            
            # Compute loss
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{Config.num_epochs}], Loss: {epoch_loss / len(data_loader):.4f}")
        
        # Save model checkpoint
        torch.save(model.state_dict(), Config.model_save_path)
        print(f"Model saved to {Config.model_save_path}")

if __name__ == "__main__":
    train()
