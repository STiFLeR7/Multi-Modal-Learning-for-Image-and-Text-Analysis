import torch
import numpy as np
import json
import os
from model import ImageCaptionModel  # Assuming the model class is defined as ImageCaptionModel
from torch.utils.data import DataLoader
from dataset import ImageCaptionDataset  # Assuming you have a Dataset class for loading data

# Path to pre-trained model
model_path = 'saved_models/model_epoch_10.pth'
# Directory for new data for inference
inference_data_dir = 'inference_data/'

# Hyperparameters (use the same ones used for training)
batch_size = 64
embedding_dim = 256

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptionModel(embedding_dim)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Prepare the DataLoader for the inference data
inference_dataset = ImageCaptionDataset(inference_data_dir)
inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

# Inference function
def infer(model, data_loader):
    predictions = []
    with torch.no_grad():
        for images, captions in data_loader:
            images = images.to(device)
            outputs = model(images)
            # Here we assume outputs are image feature vectors, you can add decoding for captions if needed
            predictions.append(outputs)
    return predictions

# Run inference
predictions = infer(model, inference_loader)
print("Inference completed. Number of predictions:", len(predictions))

# Save predictions for further analysis
with open('predictions.json', 'w') as f:
    json.dump(predictions, f)

print("Predictions saved in predictions.json.")
