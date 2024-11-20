import os
import json
from PIL import Image
import numpy as np

# Path to your images and captions
image_dir = 'D:/COCO-DATASET/coco2017/train2017'  # Adjust this path
caption_file = "D:/COCO-DATASET/coco2017/annotations/captions_train2017.json"  # Adjust this path

# Read the captions from the JSON file
with open(caption_file, 'r') as f:
    captions_data = json.load(f)

print(f"Loaded captions data with {len(captions_data['annotations'])} annotations.")

# Create a dictionary of captions for each image
captions = {}
for annotation in captions_data['annotations']:
    image_id = annotation['image_id']
    caption = annotation['caption']
    image_name = f'{image_id:012d}.jpg'  # Formatting image name to match the filenames in the directory
    if image_name not in captions:
        captions[image_name] = []
    captions[image_name].append(caption)

# Preprocessing function to load and resize images (now using float32 for optimization)
def load_image(image_path, target_size=(224, 224)):
    """
    Load an image from the given path and preprocess it (resize to target size).
    Convert grayscale images to RGB.
    """
    try:
        image = Image.open(image_path)  # Open the image using PIL
        # Convert grayscale images to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert to RGB if not already

        image = image.resize(target_size)  # Resize the image to the target size
        image = np.array(image).astype(np.float32)  # Convert to NumPy array and use float32 for memory efficiency
        image = image / 255.0  # Normalize pixel values to [0, 1]

        if image.shape == (224, 224, 3):
            return image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape} for {image_path}")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None  # Return None if the image is malformed or unreadable

def image_caption_generator(image_dir, captions, batch_size=64):
    """
    A generator function that loads images and captions in batches.
    Yields batches of images and their corresponding captions.

    Args:
    - image_dir (str): Path to the image directory.
    - captions (dict): Dictionary containing image filenames and their captions.
    - batch_size (int): Number of images and captions to yield at a time.

    Yields:
    - A tuple of (batch_images, batch_captions) where:
      - batch_images is a list of images.
      - batch_captions is a list of corresponding captions.
    """
    batch_images = []
    batch_captions = []
    
    # Shuffle the images (optional)
    image_files = list(captions.keys())
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = load_image(image_path)  # Load the image
        if image is not None:
            batch_images.append(image)
        
        # Get the captions for the current image
        image_captions = captions.get(image_file, [])
        batch_captions.extend(image_captions)
        
        # If the batch size is reached, yield the batch
        if len(batch_images) >= batch_size:
            yield np.array(batch_images), batch_captions
            batch_images = []  # Reset batch
            batch_captions = []  # Reset batch
    
    # Yield the last batch if it contains any remaining images
    if len(batch_images) > 0:
        yield np.array(batch_images), batch_captions

# Example: Use the generator to process and save data in chunks
def save_preprocessed_data_in_chunks(image_dir, captions, save_dir="preprocessed_data", batch_size=64):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Use the image_caption_generator to process in batches
    for batch_idx, (images, batch_captions) in enumerate(image_caption_generator(image_dir, captions, batch_size)):
        print(f"Processing batch {batch_idx + 1}...")

        # Ensure the images are in the correct shape (batch_size, 224, 224, 3)
        images = np.array(images)

        # Save each batch of images and captions to separate files
        np.save(os.path.join(save_dir, f"images_batch_{batch_idx + 1}.npy"), images)
        with open(os.path.join(save_dir, f"captions_batch_{batch_idx + 1}.json"), 'w') as f:
            json.dump(batch_captions, f)

    print(f"Preprocessed data saved to {save_dir}")

# Run the preprocessing function to save data in chunks
save_preprocessed_data_in_chunks(image_dir, captions)
