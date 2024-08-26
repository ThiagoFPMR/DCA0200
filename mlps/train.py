import torch
from tqdm import tqdm
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FacialEmotionDataset
from model import FacialEmotionClassifier

# Define the data directory
train_dir = '../data/FacialEmotionDataset/Train'
val_dir = '../data/FacialEmotionDataset/Val'

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create the datasets and dataloaders
train_dataset = FacialEmotionDataset(train_dir, transform=transform)
val_dataset = FacialEmotionDataset(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Create the model
model = FacialEmotionClassifier()

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a TensorBoard writer
writer = SummaryWriter()

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Moving model to {device}')
model.to(device)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Set the model to train mode
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        # Basic training loop
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    # Validation loop
    model.eval()
    running_vloss = 0.0   
    with torch.no_grad(): 
        for images, labels in tqdm(val_loader, desc=f'Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            vloss = criterion(outputs, labels)

            running_vloss += vloss.item()

    avg_loss = running_loss / len(train_loader)
    avg_vloss = running_vloss / len(val_loader)

    writer.add_scalars(
        'Loss/train vs val',
        {'Training': avg_loss, 'Validation': avg_vloss},
        epoch
    )
    writer.flush()

    # Save the model
    torch.save(model.state_dict(), 'emotion_model.pth')

writer.close()