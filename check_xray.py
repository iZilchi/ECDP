import os
from torchvision import datasets

def check_chest_xray(data_root='./data/chest_xray/chest_xray'):
    for split in ['train', 'val', 'test']:
        path = os.path.join(data_root, split)
        if os.path.exists(path):
            ds = datasets.ImageFolder(path)
            print(f"{split}: {len(ds)} images, classes: {ds.classes}")
        else:
            print(f"{split}: folder not found at {path}")

if __name__ == '__main__':
    check_chest_xray()