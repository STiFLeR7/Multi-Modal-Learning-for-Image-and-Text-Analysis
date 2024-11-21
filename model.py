import torch
import torch.nn as nn

class ImageCaptionModel(nn.Module):
    def __init__(self, embedding_dim):
        super(ImageCaptionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 112 * 112, embedding_dim)  # Example, adjust according to your input image size
        self.fc2 = nn.Linear(embedding_dim, 256)  # Adjust according to your output size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
