# setup_skin_cancer.py - FIXED VERSION
"""
Setup HAM10000 dataset for federated learning
Handles image mapping creation properly
"""
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

def create_image_mapping(data_dir):
    """Create a mapping of image IDs to their actual file paths"""
    
    print("\n🔍 Creating image mapping...")
    
    mapping = {}
    
    # Define possible image directories
    possible_dirs = [
        os.path.join(data_dir, 'HAM10000_images_part_1'),
        os.path.join(data_dir, 'HAM10000_images_part_2'),
        os.path.join(data_dir, 'images'),
        data_dir  # Also check root directory
    ]
    
    total_images = 0
    
    for img_dir in possible_dirs:
        if not os.path.exists(img_dir):
            continue
        
        print(f"  Scanning: {img_dir}")
        
        for file in os.listdir(img_dir):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Extract image ID (remove extension)
                img_id = file.replace('.jpg', '').replace('.png', '').replace('.jpeg', '').replace('.JPG', '')
                
                # Store full path
                mapping[img_id] = os.path.join(img_dir, file)
                total_images += 1
    
    # Save mapping to JSON
    mapping_path = os.path.join(data_dir, 'image_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Created mapping with {total_images} images")
    print(f"📁 Saved to: {mapping_path}")
    
    # Verify JSON is valid
    with open(mapping_path, 'r') as f:
        test_load = json.load(f)
        print(f"✅ Verified JSON is valid ({len(test_load)} entries)")
    
    return mapping

def setup_ham10000():
    """Setup HAM10000 dataset for federated learning"""
    
    print("="*70)
    print("🔧 SETTING UP HAM10000 DATASET")
    print("="*70)
    
    data_dir = './data/skin_cancer'
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    
    print("\n📁 Files in data directory:")
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            print(f"  - {file}")
    
    # Check for metadata file
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    
    if not os.path.exists(metadata_path):
        print(f"\n❌ HAM10000_metadata.csv not found!")
        print(f"📥 Please download from Kaggle and place in: {data_dir}")
        print(f"🔗 https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000")
        return False
    
    print(f"\n✅ Found metadata file: {metadata_path}")
    
    # Read metadata
    print("\n📊 Reading metadata...")
    df = pd.read_csv(metadata_path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Create image mapping FIRST
    mapping = create_image_mapping(data_dir)
    
    if len(mapping) == 0:
        print("\n❌ ERROR: No images found!")
        print("Please ensure HAM10000 images are in:")
        print("  - ./data/skin_cancer/HAM10000_images_part_1/")
        print("  - ./data/skin_cancer/HAM10000_images_part_2/")
        return False
    
    # Create train/test split
    print("\n🎯 Creating train/test split...")
    
    # Clean image_id - remove extensions if present
    df['image_id'] = df['image_id'].astype(str)
    
    # Filter to only include images we actually have
    df_filtered = df[df['image_id'].isin(mapping.keys())]
    
    print(f"  Total samples in metadata: {len(df)}")
    print(f"  Samples with images found: {len(df_filtered)}")
    
    if len(df_filtered) < 100:
        print("\n⚠️  WARNING: Very few images found!")
        print("  Please check image directories")
    
    # Create stratified split
    train_df, test_df = train_test_split(
        df_filtered, 
        test_size=0.2, 
        stratify=df_filtered['dx'], 
        random_state=42
    )
    
    # Save splits
    train_csv_path = os.path.join(data_dir, 'train.csv')
    test_csv_path = os.path.join(data_dir, 'test.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"  ✅ Saved train.csv ({len(train_df)} samples)")
    print(f"  ✅ Saved test.csv ({len(test_df)} samples)")
    
    # Print class distribution
    print("\n📊 Class distribution in training set:")
    class_dist = train_df['dx'].value_counts()
    for class_name, count in class_dist.items():
        print(f"  {class_name}: {count} samples")
    
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE!")
    print("="*70)
    print(f"📚 Training samples: {len(train_df)}")
    print(f"🧪 Test samples: {len(test_df)}")
    print(f"🖼️  Total images: {len(mapping)}")
    print("\n🎉 You can now run your experiments!")
    print("👉 Run: python experiments/run_comprehensive_comparison.py")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = setup_ham10000()
    
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        exit(1)