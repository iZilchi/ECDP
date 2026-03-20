import pandas as pd
import os
from sklearn.model_selection import train_test_split

def prepare_skin_cancer_data(data_path='./data/skin_cancer'):
    os.makedirs(data_path, exist_ok=True)
    metadata_path = os.path.join(data_path, 'HAM10000_metadata.csv')
    if not os.path.exists(metadata_path):
        print("Please download HAM10000_metadata.csv and place it in", data_path)
        return
    df = pd.read_csv(metadata_path)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    train_df.to_csv(os.path.join(data_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    print("Dataset prepared successfully!")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    return train_df, test_df

if __name__ == "__main__":
    prepare_skin_cancer_data()