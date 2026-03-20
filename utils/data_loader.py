import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image

class HAM10000Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

        mapping_file = os.path.join(data_dir, 'image_mapping.json')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.image_mapping = json.load(f)
        else:
            self.image_mapping = {}

        self.label_map = {
            'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
            'mel': 4, 'nv': 5, 'vasc': 6
        }
        self.path_cache = {}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = str(self.data_frame.iloc[idx]['image_id'])
        img_name_clean = img_name.replace('.jpg', '').replace('.JPG', '')

        image_path = self._find_image(img_name_clean, img_name)
        if image_path is None:
            image = Image.new('RGB', (224, 224), color='gray')
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224), color='gray')

        label_str = self.data_frame.iloc[idx]['dx']
        label = self.label_map[label_str]

        if self.transform:
            image = self.transform(image)
        return image, label

    def _find_image(self, img_name_clean, img_name_original):
        if img_name_clean in self.path_cache:
            return self.path_cache[img_name_clean]

        if img_name_clean in self.image_mapping:
            path = self.image_mapping[img_name_clean]
            if os.path.exists(path):
                self.path_cache[img_name_clean] = path
                return path

        possible_dirs = [
            os.path.join(self.data_dir, 'HAM10000_images_part_1'),
            os.path.join(self.data_dir, 'HAM10000_images_part_2'),
            os.path.join(self.data_dir, 'images'),
            self.data_dir
        ]
        for img_dir in possible_dirs:
            if not os.path.exists(img_dir):
                continue
            for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
                potential = os.path.join(img_dir, img_name_clean + ext)
                if os.path.exists(potential):
                    self.path_cache[img_name_clean] = potential
                    return potential
                potential2 = os.path.join(img_dir, img_name_original)
                if os.path.exists(potential2):
                    self.path_cache[img_name_clean] = potential2
                    return potential2
        return None

def dirichlet_partition(dataset, num_clients, alpha, seed=42):
    np.random.seed(seed)
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    n_classes = len(np.unique(targets))
    class_indices = [np.where(targets == c)[0] for c in range(n_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(n_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.maximum(proportions, 1e-6)
        proportions /= proportions.sum()
        indices = class_indices[c].copy()
        np.random.shuffle(indices)
        sizes = (proportions * len(indices)).astype(int)
        diff = len(indices) - sizes.sum()
        if diff > 0:
            sizes[np.argmax(sizes)] += diff
        elif diff < 0:
            sizes[np.argmin(sizes)] -= diff
        start = 0
        for i, size in enumerate(sizes):
            client_indices[i].extend(indices[start:start+size])
            start += size
    return client_indices

def get_skin_cancer_dataloaders(num_clients=3, batch_size=32, data_dir='./data/skin_cancer',
                                alpha=None, seed=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    train_dataset = HAM10000Dataset(train_csv, data_dir, transform=transform)
    test_dataset = HAM10000Dataset(test_csv, data_dir, transform=test_transform)

    if alpha is not None:
        client_indices = dirichlet_partition(train_dataset, num_clients, alpha, seed)
        client_datasets = [Subset(train_dataset, idx) for idx in client_indices]
    else:
        total_samples = len(train_dataset)
        samples_per_client = total_samples // num_clients
        client_datasets = []
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < num_clients - 1 else total_samples
            client_datasets.append(Subset(train_dataset, range(start, end)))

    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"✅ Created {num_clients} clients with Dirichlet alpha={alpha if alpha else 'IID'}")
    print(f"📚 Total training samples: {len(train_dataset)}")
    print(f"🧪 Test samples: {len(test_dataset)}")
    return client_loaders, test_loader

get_mnist_dataloaders = get_skin_cancer_dataloaders