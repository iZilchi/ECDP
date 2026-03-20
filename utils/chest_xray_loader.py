import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from .data_loader import dirichlet_partition   # reuse Dirichlet function

def get_chest_xray_dataloaders(num_clients=3, batch_size=32,
                               data_root='./data/chest_xray',
                               combine_val_test=True,
                               alpha=None, seed=42):
    """
    Returns client loaders and test loader for Chest X‑Ray dataset.
    - If alpha is None: IID sequential split.
    - If alpha is a float: Dirichlet(alpha) non‑IID split.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    base_dir = os.path.join(data_root, 'chest_xray')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    if combine_val_test:
        test_dataset = torch.utils.data.ConcatDataset([val_dataset, test_dataset])

    # Partition training set
    if alpha is not None:
        client_indices = dirichlet_partition(full_train_dataset, num_clients, alpha, seed)
        client_datasets = [Subset(full_train_dataset, idx) for idx in client_indices]
    else:
        total_samples = len(full_train_dataset)
        samples_per_client = total_samples // num_clients
        client_datasets = []
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < num_clients - 1 else total_samples
            client_datasets.append(Subset(full_train_dataset, range(start, end)))

    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"✅ Created {num_clients} clients with Dirichlet alpha={alpha if alpha else 'IID'}")
    print(f"📚 Total training samples: {len(full_train_dataset)}")
    print(f"🧪 Test samples: {len(test_dataset)}")
    return client_loaders, test_loader