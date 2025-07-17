import os
import cv2
import shutil
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def copy_files_parallel(paths, labels, destination_folder):
    def copy_file(path, dest):
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(path, dest)

    with ThreadPoolExecutor() as executor:
        futures = []
        for path, label in zip(paths, labels):
            label_folder = "real" if label == 0 else "fake"
            dest = os.path.join(destination_folder, label_folder, os.path.basename(path))
            futures.append(executor.submit(copy_file, path, dest))
        for future in futures:
            future.result()


def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
