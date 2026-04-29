import os

def find_chest_xray_root(base_dir):
    """Find directory containing train/test/val subfolders."""
    required = ['train', 'test', 'val']
    if all(os.path.isdir(os.path.join(base_dir, d)) for d in required):
        return base_dir
    for item in os.listdir(base_dir):
        candidate = os.path.join(base_dir, item)
        if os.path.isdir(candidate):
            if all(os.path.isdir(os.path.join(candidate, d)) for d in required):
                return candidate
    return None

def prepare_chest_xray_data(data_dir='./data/chest_xray'):
    """Verify Chest X-ray dataset structure (handles nested folder)."""
    print("="*70)
    print("🔧 SETTING UP CHEST X-RAY DATASET")
    print("="*70)
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Find actual dataset root
    root = find_chest_xray_root(data_dir)
    if root is None:
        print(f"⚠️ Could not find train/test/val directories under {data_dir}")
        print("Please ensure your Chest X-ray dataset has the following structure:")
        print(f"{data_dir}/")
        print("  ├── chest_xray/         (or directly)")
        print("  │   ├── train/")
        print("  │   │   ├── NORMAL/")
        print("  │   │   └── PNEUMONIA/")
        print("  │   ├── test/")
        print("  │   │   ├── NORMAL/")
        print("  │   │   └── PNEUMONIA/")
        print("  │   └── val/")
        print("  │       ├── NORMAL/")
        print("  │       └── PNEUMONIA/")
        return False
    
    print(f"✅ Found dataset at: {root}")
    
    # Count samples
    total_train = 0
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(root, 'train', class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            total_train += count
            print(f"Train {class_name}: {count} images")
    
    total_test = 0
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(root, 'test', class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            total_test += count
            print(f"Test {class_name}: {count} images")
    
    print(f"\n✅ Dataset ready: {total_train} training, {total_test} test samples")
    return True

if __name__ == "__main__":
    prepare_chest_xray_data()
