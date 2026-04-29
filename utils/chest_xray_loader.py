import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image

def find_chest_xray_root(base_dir):
    """
    Find the directory that contains train/, test/, val/ subfolders.
    Searches base_dir and one level deeper.
    """
    # Check base_dir itself
    required = ['train', 'test', 'val']
    if all(os.path.isdir(os.path.join(base_dir, d)) for d in required):
        return base_dir
    # Check one level deeper
    for item in os.listdir(base_dir):
        candidate = os.path.join(base_dir, item)
        if os.path.isdir(candidate):
            if all(os.path.isdir(os.path.join(candidate, d)) for d in required):
                return candidate
    raise FileNotFoundError(f"Could not find train/test/val directories under {base_dir}")

class ChestXrayDataset(Dataset):
    """Custom dataset for Chest X-ray pneumonia images."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_chest_xray_dataloaders(num_clients=3, batch_size=32, data_dir='./data/chest_xray',
                               iid=True, dirichlet_alpha=0.5, seed=42):
    """
    Create federated dataloaders for Chest X-ray (pneumonia) dataset.
    Automatically finds nested structure if needed.
    """
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Find the actual dataset root
    root = find_chest_xray_root(data_dir)
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'test')
    # Use test for validation if val missing (but we expect val)
    val_dir = os.path.join(root, 'val')
    if not os.path.exists(val_dir):
        val_dir = test_dir
    
    train_dataset = ChestXrayDataset(train_dir, transform=transform_train)
    test_dataset = ChestXrayDataset(test_dir, transform=transform_test)
    
    # Get labels for all training samples
    labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    
    # Shuffle all indices first
    np.random.seed(seed)
    all_indices = np.random.permutation(len(train_dataset))
    
    if iid:
        samples_per_client = len(train_dataset) // num_clients
        client_indices = []
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < num_clients - 1 else len(train_dataset)
            client_indices.append(all_indices[start:end].tolist())
    else:
        # Non-IID: Dirichlet distribution over classes (2 classes)
        num_classes = 2
        client_indices = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_mask = (labels[all_indices] == class_id)
            class_indices = all_indices[class_mask]
            if len(class_indices) == 0:
                continue
            proportions = np.random.dirichlet([dirichlet_alpha] * num_clients)
            split_sizes = (proportions * len(class_indices)).astype(int)
            split_sizes[-1] = len(class_indices) - sum(split_sizes[:-1])
            start = 0
            for client_id, size in enumerate(split_sizes):
                if size > 0:
                    client_indices[client_id].extend(class_indices[start:start+size].tolist())
                start += size
        # Shuffle each client's indices
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
    
    client_datasets = [Subset(train_dataset, indices) for indices in client_indices]
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
