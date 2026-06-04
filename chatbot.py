"""
Week 1 - NN and CNN

Simple implementation for FashionMNIST.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

print("Device:", device)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted([f"{i:03d}.png" for i in range(100)])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, img_name[:3]

testset = TestDataset("./images", transform=transform)
testloader = DataLoader(testset, batch_size=10, shuffle=False)

# MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)

# CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Training
def train_model(model, epochs, name):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0

        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"{name} | Epoch {epoch+1}/{epochs} | Loss: {running_loss:.3f}")

    return model

# Submission
def generate_submission(model, filename):
    model.eval()

    preds = []

    with torch.no_grad():
        for images, img_ids in testloader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for img_id, pred in zip(img_ids, predicted):
                preds.append({
                    "image_id": img_id,
                    "label": pred.item()
                })

    pd.DataFrame(preds).to_csv(filename, index=False)
    print("Saved:", filename)

epochs = 5

mlp_model = train_model(MLP(), epochs, "MLP")
generate_submission(mlp_model, "2025TT11556_NN.csv")

cnn_model = train_model(CNN(), epochs, "CNN")
generate_submission(cnn_model, "2025TT11556_CNN.csv")

print("Done")
