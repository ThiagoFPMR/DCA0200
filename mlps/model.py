import timm
import torch.nn as nn

# PyTorch models inherit from torch.nn.Module
class FacialEmotionClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(FacialEmotionClassifier, self).__init__()
        # Where we define all the parts of the model
        self.classifier = nn.Sequential(
            nn.Linear(128*128, 64*64),
            nn.ReLU(),
            nn.Linear(64*64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        # Turn image to grayscale
        x = x.mean(dim=1)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output