import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from train import ImageCaptionDataset, ImageCaptionModel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
data_dir = 'preprocessed_data'  # Directory with validation/test data
model_dir = 'saved_models'      # Directory where models are saved

# Hyperparameters
batch_size = 64

# Find the latest model or prompt the user
def get_model_path():
    model_files = [f for f in os.listdir(model_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"No model files found in directory: {model_dir}")
    
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by epoch number
    print("Available models:", model_files)
    
    # Automatically select the latest model
    latest_model = model_files[-1]
    print(f"Using the latest model: {latest_model}")
    
    # Uncomment this to allow manual selection:
    # for idx, model_file in enumerate(model_files):
    #     print(f"{idx + 1}. {model_file}")
    # choice = int(input("Select the model to evaluate (number): ")) - 1
    # return os.path.join(model_dir, model_files[choice])
    
    return os.path.join(model_dir, latest_model)

# Load the saved model
def load_model(model_path):
    model = ImageCaptionModel(embedding_dim=256)  # Use the same embedding_dim as in train.py
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

# Evaluation function
def evaluate_model():
    # Load data
    dataset = ImageCaptionDataset(data_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get model path and load the model
    model_path = get_model_path()
    model = load_model(model_path)
    
    total_loss = 0.0
    total_samples = 0
    all_outputs = []
    all_captions = []
    criterion = torch.nn.CrossEntropyLoss()  # Same loss function as in training

    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(data_loader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Example: Convert captions to tensor indices (requires tokenizer logic for real cases)
            # For simplicity, using dummy targets with zero indices (adjust this for real evaluation)
            targets = torch.zeros(outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            all_outputs.extend(outputs.cpu().numpy())
            all_captions.extend(captions)

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")

    avg_loss = total_loss / total_samples
    print(f"Evaluation Complete! Average Loss: {avg_loss:.4f}")
    
    # Optionally, return metrics or predictions
    return avg_loss, all_outputs, all_captions

# Main entry point
if __name__ == "__main__":
    avg_loss, outputs, captions = evaluate_model()
    print("Evaluation Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Sample Outputs: {outputs[:5]}")
    print(f"Sample Captions: {captions[:5]}")
