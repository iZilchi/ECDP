# utils/prepare_skin_cancer_data.py - NEW FILE
"""
Script to prepare HAM10000 dataset for federated learning
Download from: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
"""
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_skin_cancer_data(data_path='./data/skin_cancer'):
    """Prepare HAM10000 dataset for federated learning experiments"""
    
    # Create directories
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.join(data_path, 'images'), exist_ok=True)
    
    # Assuming you have downloaded the dataset and have:
    # - HAM10000_metadata.csv
    # - images in a folder
    
    # Read metadata
    metadata_path = os.path.join(data_path, 'HAM10000_metadata.csv')
    if not os.path.exists(metadata_path):
        print("Please download HAM10000_metadata.csv and place it in", data_path)
        return
    
    df = pd.read_csv(metadata_path)
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    
    # Save splits
    train_df.to_csv(os.path.join(data_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    
    print("Dataset prepared successfully!")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print("Class distribution in training set:")
    print(train_df['dx'].value_counts())
    
    return train_df, test_df

if __name__ == "__main__":
    prepare_skin_cancer_data()