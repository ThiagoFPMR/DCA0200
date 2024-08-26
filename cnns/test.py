"""Inference code for the ASL model."""

import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import ASLDataset
from model import ASLClassifier

# Define the input image path
image_path = '../data/testing_data/istockphoto-960235416-612x612.jpg'

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the image
image = transform(Image.open(image_path)).unsqueeze(0)
dataset = ASLDataset('../data/ASL_Dataset/Test', transform=transform)

# Load the model
model = ASLClassifier()
model.load_state_dict(torch.load('weights/asl_model.pth'))
model.eval()

# Make a prediction
output = model(image)
prediction = torch.argmax(output, dim=1).item()
print(f'Predicted class: {dataset.classes[prediction]}')

