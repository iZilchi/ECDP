# check_dataset.py - DIAGNOSTIC TOOL
"""
Quick diagnostic to check HAM10000 dataset status
Run this to see what's wrong with your setup
"""
import os
import json
import pandas as pd

def check_dataset():
    """Comprehensive dataset check"""
    
    print("="*70)
    print("🔍 HAM10000 DATASET DIAGNOSTIC")
    print("="*70)
    
    data_dir = './data/skin_cancer'
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print(f"👉 Create it: mkdir -p {data_dir}")
        return False
    
    print(f"✅ Data directory exists: {data_dir}\n")
    
    # List all files in data directory
    print("📁 Files in data directory:")
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Show first 10 files per directory
            print(f'{subindent}{file}')
            all_files.append(os.path.join(root, file))
        if len(files) > 10:
            print(f'{subindent}... and {len(files)-10} more files')
    
    print(f"\n📊 Total files found: {len(all_files)}\n")
    
    # Check for metadata CSV
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    if os.path.exists(metadata_path):
        print(f"✅ Found metadata: {metadata_path}")
        df = pd.read_csv(metadata_path)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
    else:
        print(f"❌ Metadata not found: {metadata_path}")
        print(f"👉 Download from: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000")
    
    print()
    
    # Check for train/test CSVs
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    
    if os.path.exists(train_csv):
        print(f"✅ Found train.csv ({len(pd.read_csv(train_csv))} samples)")
    else:
        print(f"❌ train.csv not found")
        print(f"👉 Run: python setup_skin_cancer.py")
    
    if os.path.exists(test_csv):
        print(f"✅ Found test.csv ({len(pd.read_csv(test_csv))} samples)")
    else:
        print(f"❌ test.csv not found")
        print(f"👉 Run: python setup_skin_cancer.py")
    
    print()
    
    # Check for image directories
    image_dirs = [
        'HAM10000_images_part_1',
        'HAM10000_images_part_2',
        'images'
    ]
    
    total_images = 0
    for dir_name in image_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"✅ Found {dir_name}: {len(images)} images")
            total_images += len(images)
        else:
            print(f"⚠️  {dir_name} not found")
    
    print(f"\n🖼️  Total images found: {total_images}\n")
    
    # Check image mapping
    mapping_path = os.path.join(data_dir, 'image_mapping.json')
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                content = f.read()
                if content.strip():
                    mapping = json.loads(content)
                    print(f"✅ Image mapping exists: {len(mapping)} entries")
                else:
                    print(f"❌ Image mapping file is EMPTY")
                    print(f"👉 Run: python setup_skin_cancer.py")
        except json.JSONDecodeError as e:
            print(f"❌ Image mapping is CORRUPTED: {e}")
            print(f"👉 Run: python setup_skin_cancer.py")
    else:
        print(f"⚠️  Image mapping not found")
        print(f"👉 Run: python setup_skin_cancer.py")
    
    print("\n" + "="*70)
    
    # Final recommendation
    if total_images == 0:
        print("❌ CRITICAL: No images found!")
        print("\n📥 Download HAM10000 dataset from Kaggle:")
        print("   https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000")
        print(f"\n📁 Extract to: {data_dir}")
        print("   Expected structure:")
        print("   ./data/skin_cancer/")
        print("   ├── HAM10000_metadata.csv")
        print("   ├── HAM10000_images_part_1/")
        print("   │   └── ISIC_*.jpg")
        print("   └── HAM10000_images_part_2/")
        print("       └── ISIC_*.jpg")
        return False
    
    elif not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("⚠️  Dataset files not processed yet")
        print("\n👉 Next step: Run setup script")
        print("   python setup_skin_cancer.py")
        return False
    
    else:
        print("✅ DATASET READY!")
        print("\n👉 You can now run experiments:")
        print("   python experiments/run_comprehensive_comparison.py")
        return True
    
    print("="*70)

if __name__ == "__main__":
    check_dataset()