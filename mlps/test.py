"""Inference code for the ASL model."""

import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import FacialEmotionDataset
from model import FacialEmotionClassifier

# Define the input image path
image_path = '../data/FacialEmotionDataset/Val/Happy/5_Happy.jpg'

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the image
image = transform(Image.open(image_path)).unsqueeze(0)
dataset = FacialEmotionDataset('../data/FacialEmotionDataset/Val', transform=transform)

# Load the model
model = FacialEmotionClassifier()
model.load_state_dict(torch.load('emotion_model.pth'))
model.eval()

# Make a prediction
output = model(image)
prediction = torch.argmax(output, dim=1).item()
print(f'Predicted class: {dataset.classes[prediction]}')

