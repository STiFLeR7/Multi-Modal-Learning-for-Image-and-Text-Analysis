import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

# Define your model architecture (Example: ResNet)
model = models.resnet50(pretrained=False)  # Set pretrained=False to load custom weights

# Load the saved model weights
model.load_state_dict(torch.load('D:/Multi-Modal-Learning-for-Image-and-Text-Analysis/saved_models/model_epoch_10.pth'))

# Set the model to evaluation mode
model.eval()

# Define a transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size the model expects
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Receive an image from the request
    file = request.files['image']
    image = Image.open(file.stream)

    # Apply the transformation to the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the predicted class

    # Return the predicted class as a response
    return jsonify({'prediction': predicted.item()})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
