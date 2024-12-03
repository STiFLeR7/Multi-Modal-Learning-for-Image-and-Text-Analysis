import json
import os
import matplotlib.pyplot as plt
from PIL import Image

# Paths
OUTPUT_FILE = './predictions.json'  # Path to your predictions file
COCO_DATASET_DIR = './preprocessed_data/'  # Adjust if needed

# Load predictions from JSON file
def load_predictions(file_path):
    with open(file_path, 'r') as f:
        predictions = json.load(f)
    return predictions

# Visualize predictions
def visualize_predictions(predictions, dataset_dir):
    for prediction in predictions:
        # Extract information
        image_id = prediction.get('image_id')
        predicted_class = prediction.get('predicted_class')
        confidence = prediction.get('confidence')
        
        # Skip invalid entries
        if not image_id or not predicted_class or confidence is None:
            continue
        
        # Define image path based on the structure of your dataset
        image_path = os.path.join(dataset_dir, 'images_batch', f'image_{image_id}.jpg')  # Adjust path as needed

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Open and display the image
        image = Image.open(image_path)
        
        # Show the image with the predicted class and confidence
        plt.imshow(image)
        plt.title(f'Predicted: {predicted_class}, Confidence: {confidence:.4f}')
        plt.axis('off')  # Hide axes
        plt.show()

# Main script
if __name__ == '__main__':
    # Load predictions from JSON file
    predictions = load_predictions(OUTPUT_FILE)

    if len(predictions) == 0:
        print("No predictions found in the JSON file.")
    else:
        # Visualize predictions
        visualize_predictions(predictions, COCO_DATASET_DIR)
