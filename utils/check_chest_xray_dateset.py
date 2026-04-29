import os

def find_chest_xray_root(base_dir):
    required = ['train', 'test', 'val']
    if all(os.path.isdir(os.path.join(base_dir, d)) for d in required):
        return base_dir
    for item in os.listdir(base_dir):
        candidate = os.path.join(base_dir, item)
        if os.path.isdir(candidate):
            if all(os.path.isdir(os.path.join(candidate, d)) for d in required):
                return candidate
    return None

def check_chest_xray_dataset():
    print("="*70)
    print("🔍 CHEST X-RAY DATASET DIAGNOSTIC")
    print("="*70)
    
    data_dir = './data/chest_xray'
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    root = find_chest_xray_root(data_dir)
    if root is None:
        print(f"❌ Could not find train/test/val directories under {data_dir}")
        return False
    
    print(f"✅ Found dataset at: {root}")
    for d in ['train', 'test', 'val']:
        full = os.path.join(root, d)
        print(f"✅ {full} exists")
    
    print("\n✅ DATASET READY!")
    print("👉 You can now run experiments with --dataset chest")
    return True

if __name__ == "__main__":
    check_chest_xray_dataset()
