
# Multi-Modal Learning for Image and Text Analysis

![Banner]("https://github.com/STiFLeR7/Multi-Modal-Learning-for-Image-and-Text-Analysis/blob/master/banner.jpeg")
**Overview**

This repository focuses on multi-modal learning, integrating image processing and natural language understanding. The project aims to generate descriptive captions for images using a combination of deep learning techniques for vision-language models.

**Features**

End-to-end text-to-image and image-to-text generation.

Use of ResNet18 for image feature extraction.

A custom Transformer-based model for text generation.

Extensive data augmentation to improve model generalization.

Evaluation metrics: BLEU Score, Cross-Entropy Loss, and Accuracy.


## Table of Contents

1. Project Architecture

2. Dataset

3. Setup Instructions

4. Training and Validation

5. Results

6. Future Improvements

7. Contributors

## Project Architecture

The project architecture consists of the following components:

1. **Image Encoder**:
    
    Pretrained ResNet18 extracts visual features from images.

    Features are projected into an embedding space of dimension ```embedding_dim```.

2. **Text Encoder**:
    
    Embeds the captions into tokenized feature vectors.
    
    Custom Embedding Layer maps vocabulary tokens into the same embedding space as the image features.

3. **Fusion Layer**:

    Combines image and text embeddings for feature fusion.
    
    Fully connected layers integrate both modalities.

4. **Output Decoder**:

    Generates a sequence of tokens as text captions.

    Evaluated using Cross-Entropy Loss and **BLEU Score**.
## Dataset


This project uses the **Flickr8k** dataset:

**Image Directory**: Contains 8,000 images.
    
**Caption File**: Each image is annotated with five captions.
    
**Augmentation**: Images are augmented with random flips, rotations, and color jittering to increase dataset variability.
## Setup Instructions

1. **Clone this Repository**
```git clone https://github.com/STiFLeR7/Multi-Modal-Learning-for-Image-and-Text-Analysis```

```cd Multi-Modal-Learning-for-Image-and-Text-Analysis```

2. **Install Dependencies**
```pip install -r requirements.txt```

3. **Running the Python Files**
Run Data Augmentation - ```python augmented.py``` 

Train the Model - ```python train.py```

Validate the Model - ```python validate.py```
## Training and Validation

**Training**

The training process involves:

    1. Cross-Entropy Loss for token predictions.
    2. Gradient Clipping to prevent exploding gradients.
    3. Checkpointing to save the best model based on validation loss.

**Validation**

Evaluation metrics include:

    1. Validation Loss: Monitors overfitting.
    2. BLEU Score: Evaluates sequence-to-sequence quality.
    3. Accuracy: Measures token-level predictions.
## Results





**Metrics**

Training Loss: 4.5278

Validation BLEU Score: 0.6543

Validation Accuracy: 83.45%
## Future Improvements

    1. Implement Transformer-based decoders for more accurate caption generation.

    2. Experiment with larger datasets like COCO for better generalization.
    
    3. Add Beam Search Decoding for generating captions.
## Contributors

**STiFLeR7** - Lead Developer
