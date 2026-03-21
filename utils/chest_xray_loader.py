import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

def get_chest_xray_dataloaders(
    num_clients=10,
    batch_size=64,
    data_root='./data/chest_xray',
    combine_val_test=True   # combine val and test into one test set
):
    """
    Returns client loaders (for training) and a test loader.
    Assumes the dataset is at: data_root/chest_xray/ (with train, val, test subfolders)
    """
    # Transforms: resize to 224x224 (same as model expects)
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

    # Paths
    base_dir = os.path.join(data_root, 'chest_xray')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Load full training set
    full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # Load validation and test sets
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Combine val and test if desired
    if combine_val_test:
        test_dataset = torch.utils.data.ConcatDataset([val_dataset, test_dataset])
    else:
        test_dataset = test_dataset   # keep only test folder as test

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Split the training set into client subsets
    total_samples = len(full_train_dataset)
    samples_per_client = total_samples // num_clients
    client_datasets = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else total_samples
        client_datasets.append(Subset(full_train_dataset, range(start, end)))

    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]

    print(f"✅ Created {num_clients} clients with ~{samples_per_client} samples each")
    print(f"📚 Total training samples: {total_samples}")
    print(f"🧪 Test samples (val+test): {len(test_dataset)}")
    return client_loaders, test_loader