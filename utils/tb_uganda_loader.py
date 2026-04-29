# utils/tb_uganda_loader.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class TBUgandaDataset(Dataset):
    def __init__(self, tb_dir, normal_dir, transform=None):
        self.transform = transform
        self.samples = []

        # Load TB-positive images
        tb_images = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Load normal (negative) images from nested pneumonia dataset path
        normal_images = []
        if os.path.exists(normal_dir):
            normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # Match the number of TB images
            if len(normal_images) > len(tb_images):
                normal_images = normal_images[:len(tb_images)]
        else:
            raise FileNotFoundError(f"Normal images directory not found at {normal_dir}")

        # Assign labels: TB = 1, Normal = 0
        for img in tb_images:
            self.samples.append((img, 1))
        for img in normal_images:
            self.samples.append((img, 0))

        # Shuffle
        np.random.seed(42)
        np.random.shuffle(self.samples)

        print(f"Loaded dataset: {len(tb_images)} TB, {len(normal_images)} Normal (total {len(self.samples)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_tb_uganda_dataloaders(num_clients=5, batch_size=32, data_dir='./data',
                              iid=True, dirichlet_alpha=0.5, seed=42):
    """
    Creates federated dataloaders for Uganda TB dataset.
    Images are resized to 32x32 for faster training.
    """
    # Transforms with 32x32 output
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),          # <-- reduced from 224 to 32
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),          # <-- same
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    tb_dir = os.path.join(data_dir, 'chest_xray_tb')
    # Nested path from your system
    normal_dir = os.path.join(data_dir, 'chest_xray', 'chest_xray', 'train', 'NORMAL')

    if not os.path.exists(tb_dir):
        raise FileNotFoundError(f"TB images directory not found: {tb_dir}")
    if not os.path.exists(normal_dir):
        raise FileNotFoundError(f"Normal images directory not found: {normal_dir}")

    full_dataset = TBUgandaDataset(tb_dir, normal_dir, transform=train_transform)

    # Train/test split (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    test_dataset.dataset.transform = test_transform

    # Federated partitioning
    np.random.seed(seed)
    all_indices = np.random.permutation(train_size)
    samples_per_client = train_size // num_clients
    client_indices = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else train_size
        client_indices.append(all_indices[start:end].tolist())

    client_datasets = [torch.utils.data.Subset(train_dataset, idx) for idx in client_indices]
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nCreated {num_clients} clients for Uganda TB dataset (32x32 images).")
    print(f"  Training samples: {train_size}")
    print(f"  Test samples: {test_size}")
    for i, loader in enumerate(client_loaders):
        labels = [label for _, label in loader.dataset]
        print(f"  Client {i+1}: {labels.count(1)} TB, {labels.count(0)} Normal")

    return client_loaders, test_loader
