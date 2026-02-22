# utils/data_loader.py - FIXED VERSION WITH ROBUST ERROR HANDLING
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
from PIL import Image
import os
import torch
import json

class HAM10000Dataset(torch.utils.data.Dataset):
    """Custom dataset for HAM10000 skin cancer images - ROBUST VERSION"""
    
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        
        # Load image mapping with error handling
        mapping_file = os.path.join(data_dir, 'image_mapping.json')
        
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    content = f.read()
                    if content.strip():  # Check if file is not empty
                        self.image_mapping = json.loads(content)
                        print(f"✅ Loaded image mapping with {len(self.image_mapping)} entries")
                    else:
                        print("⚠️  Image mapping file is empty, will search for images")
                        self.image_mapping = {}
            except json.JSONDecodeError as e:
                print(f"⚠️  Error reading image mapping: {e}")
                print("⚠️  Will search for images directly")
                self.image_mapping = {}
        else:
            print("⚠️  Image mapping not found, will search for images")
            self.image_mapping = {}
        
        # Map string labels to integers
        self.label_map = {
            'akiec': 0,  # Actinic keratoses
            'bcc': 1,    # Basal cell carcinoma
            'bkl': 2,    # Benign keratosis-like lesions
            'df': 3,     # Dermatofibroma
            'mel': 4,    # Melanoma
            'nv': 5,     # Melanocytic nevi
            'vasc': 6    # Vascular lesions
        }
        
        # Cache for found image paths
        self.path_cache = {}
        
        print(f"📚 Loaded {len(self.data_frame)} samples from {csv_file}")
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx]['image_id']
        
        # Clean the image name (remove extension if present)
        img_name_clean = str(img_name).replace('.jpg', '').replace('.JPG', '')
        
        # Try to find the image
        image_path = self._find_image(img_name_clean, img_name)
        
        if image_path is None:
            # Use a placeholder image if not found
            print(f"⚠️  Image not found: {img_name}, using placeholder")
            image = Image.new('RGB', (28, 28), color='gray')
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"⚠️  Error loading {image_path}: {e}, using placeholder")
                image = Image.new('RGB', (28, 28), color='gray')
        
        label_str = self.data_frame.iloc[idx]['dx']
        label = self.label_map[label_str]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _find_image(self, img_name_clean, img_name_original):
        """Find image path using multiple strategies"""
        
        # Strategy 1: Check cache
        if img_name_clean in self.path_cache:
            return self.path_cache[img_name_clean]
        
        # Strategy 2: Use mapping
        if img_name_clean in self.image_mapping:
            path = self.image_mapping[img_name_clean]
            if os.path.exists(path):
                self.path_cache[img_name_clean] = path
                return path
        
        # Strategy 3: Search in common directories
        possible_dirs = [
            os.path.join(self.data_dir, 'HAM10000_images_part_1'),
            os.path.join(self.data_dir, 'HAM10000_images_part_2'),
            os.path.join(self.data_dir, 'images'),
            self.data_dir
        ]
        
        for img_dir in possible_dirs:
            if not os.path.exists(img_dir):
                continue
            
            # Try different extensions
            for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
                # Try with cleaned name
                potential_path = os.path.join(img_dir, img_name_clean + ext)
                if os.path.exists(potential_path):
                    self.path_cache[img_name_clean] = potential_path
                    return potential_path
                
                # Try with original name
                potential_path = os.path.join(img_dir, str(img_name_original))
                if os.path.exists(potential_path):
                    self.path_cache[img_name_clean] = potential_path
                    return potential_path
        
        return None

def get_skin_cancer_dataloaders(num_clients=3, batch_size=64, data_dir='./data/skin_cancer'):
    """Create HAM10000 dataloaders for federated learning"""
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    
    # Check if files exist
    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"❌ Train CSV not found at {train_csv}\n"
            f"👉 Run: python setup_skin_cancer.py"
        )
    
    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"❌ Test CSV not found at {test_csv}\n"
            f"👉 Run: python setup_skin_cancer.py"
        )
    
    print(f"\n📂 Loading datasets from {data_dir}")
    
    train_dataset = HAM10000Dataset(train_csv, data_dir, transform=transform)
    test_dataset = HAM10000Dataset(test_csv, data_dir, transform=test_transform)
    
    # Split data among clients
    total_samples = len(train_dataset)
    samples_per_client = total_samples // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else total_samples
        client_data = Subset(train_dataset, range(start_idx, end_idx))
        client_datasets.append(client_data)
    
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Created {num_clients} clients with ~{samples_per_client} samples each")
    print(f"📚 Total training samples: {total_samples}")
    print(f"🧪 Test samples: {len(test_dataset)}\n")
    
    return client_loaders, test_loader

# For backward compatibility
get_mnist_dataloaders = get_skin_cancer_dataloaders