import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

# --- Config ---
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.001
IMG_SIZE = 64  # Small 3D volume for sample
NUM_CLASSES = 2

# --- Model Definition ---
class SimpleMRI_CNN(nn.Module):
    def __init__(self):
        super(SimpleMRI_CNN, self).__init__()
        # Input: 1 channel (grayscale MRI), 64x64x64
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        # after pool 1: 32x32x32
        # after pool 2: 16x16x16
        self.fc1 = nn.Linear(16 * 16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Synthetic Dataset ---
class SyntheticMRIDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random 3D noise + simplified shapes
        # Class 0: Noise
        # Class 1: Noise + "Tumor" (bright spot)
        label = np.random.randint(0, 2)
        data = np.random.normal(0, 0.5, (1, IMG_SIZE, IMG_SIZE, IMG_SIZE))
        
        if label == 1:
            # Add a "tumor"
            center = np.random.randint(10, IMG_SIZE-10, 3)
            r = 5
            x, y, z = np.ogrid[:IMG_SIZE, :IMG_SIZE, :IMG_SIZE]
            mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= r**2
            data[0][mask] += 2.0 # Bright spot
            
        return torch.FloatTensor(data), label

# --- Training Loop ---
def train():
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleMRI_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    dataset = SyntheticMRIDataset(size=50) # Small dataset for fast sample run
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")
        
    print(f"Training complete in {time.time() - start_time:.2f}s")
    
    # Save model
    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), "output/mri_cnn_model.pth")
    print("Model saved to output/mri_cnn_model.pth")

if __name__ == "__main__":
    train()
