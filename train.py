import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from dataset_utils import CustomImageDataset, copy_files_parallel, get_transforms
from model import ModifiedMobileNetV2

# Define paths
real_path = "/content/drive/MyDrive/Data/raw_images/real"
fake_path = "/content/drive/MyDrive/Data/raw_images/fake"
train_dir = "/content/drive/MyDrive/Data/image_dataset/train"
val_dir = "/content/drive/MyDrive/Data/image_dataset/validation"
test_dir = "/content/drive/MyDrive/Data/image_dataset/test"

# Load image paths and labels
real_images = [(os.path.join(real_path, img), 0) for img in os.listdir(real_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
fake_images = [(os.path.join(fake_path, img), 1) for img in os.listdir(fake_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
all_images = real_images + fake_images
paths, labels = zip(*all_images)

# Train-validation-test split
train_paths, temp_paths, train_labels, temp_labels = train_test_split(paths, labels, test_size=0.3, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42)

# Copy data
copy_files_parallel(train_paths, train_labels, train_dir)
copy_files_parallel(val_paths, val_labels, val_dir)
copy_files_parallel(test_paths, test_labels, test_dir)

# Prepare datasets
transform = get_transforms()
train_dataset = CustomImageDataset(train_paths, train_labels, transform=transform)
val_dataset = CustomImageDataset(val_paths, val_labels, transform=transform)
test_dataset = CustomImageDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModifiedMobileNetV2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scaler = torch.cuda.amp.GradScaler()

# Train function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    best_val_loss = float('inf')
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/content/drive/MyDrive/Data/image_classification_model.pth")

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
    print("Training complete!")
