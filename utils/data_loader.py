from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
from PIL import Image
import os
import torch
import json
import numpy as np

class HAM10000Dataset(torch.utils.data.Dataset):
    """Custom dataset for HAM10000 skin cancer images."""
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
            image = Image.new('RGB', (28, 28), color='gray')
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                image = Image.new('RGB', (28, 28), color='gray')

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

def get_skin_cancer_dataloaders(num_clients=3, batch_size=64, data_dir='./data/skin_cancer',
                                iid=True, dirichlet_alpha=0.5, seed=42):
    """
    Create federated dataloaders with IID or non-IID (Dirichlet) partitioning.

    Args:
        num_clients: number of clients
        batch_size: batch size for each client
        data_dir: path to dataset
        iid: if True, split data evenly (IID). If False, use Dirichlet distribution for non-IID.
        dirichlet_alpha: concentration parameter (lower = more heterogeneous)
        seed: random seed for reproducibility
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    train_dataset = HAM10000Dataset(train_csv, data_dir, transform=transform)
    test_dataset = HAM10000Dataset(test_csv, data_dir, transform=test_transform)

    # Get labels for all training samples
    labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    labels = np.array(labels)

    # CRITICAL FIX: Shuffle all indices first to ensure IID split is truly random
    np.random.seed(seed)
    all_indices = np.random.permutation(len(train_dataset))

    if iid:
        # IID: split shuffled indices evenly
        samples_per_client = len(train_dataset) // num_clients
        client_indices = []
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < num_clients - 1 else len(train_dataset)
            client_indices.append(all_indices[start:end].tolist())
    else:
        # Non-IID: Dirichlet distribution over classes using shuffled indices
        np.random.seed(seed)
        num_classes = 7
        client_indices = [[] for _ in range(num_clients)]

        # For each class, split samples among clients using Dirichlet
        for class_id in range(num_classes):
            # Get indices of this class from the SHUFFLED list
            class_mask = (labels[all_indices] == class_id)
            class_indices = all_indices[class_mask]

            if len(class_indices) == 0:
                continue

            # Draw proportions from Dirichlet
            proportions = np.random.dirichlet([dirichlet_alpha] * num_clients)
            # Calculate split sizes
            split_sizes = (proportions * len(class_indices)).astype(int)
            # Ensure sum matches exactly
            split_sizes[-1] = len(class_indices) - sum(split_sizes[:-1])

            start = 0
            for client_id, size in enumerate(split_sizes):
                if size > 0:
                    client_indices[client_id].extend(class_indices[start:start + size].tolist())
                start += size

        # Shuffle each client's indices to avoid order bias
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])

    # Create Subset datasets and loaders
    client_datasets = [Subset(train_dataset, indices) for indices in client_indices]
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print distribution summary
    print(f"✅ Created {num_clients} clients with {'IID' if iid else f'non-IID (α={dirichlet_alpha})'} distribution")
    print(f"📚 Total training samples: {len(train_dataset)}")
    print(f"🧪 Test samples: {len(test_dataset)}")
    if not iid:
        print("Class distribution per client (class counts):")
        for i, indices in enumerate(client_indices):
            client_labels = [labels[idx] for idx in indices]
            unique, counts = np.unique(client_labels, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"  Client {i+1}: {dist}")

    return client_loaders, test_loader

# Alias for backward compatibility
get_mnist_dataloaders = get_skin_cancer_dataloaders