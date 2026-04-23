"""
Analyze model update norms after local training.
Now supports skin (HAM10000), chest X-ray (pneumonia), and TB Uganda datasets.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import argparse
from utils.data_loader import get_skin_cancer_dataloaders
from utils.chest_xray_loader import get_chest_xray_dataloaders
from utils.tb_uganda_loader import get_tb_uganda_dataloaders
from models.medium_cnn import MediumCNN
from models.chest_cnn import ChestCNN

# Try to import TBCNN, if not available, define a simple one
try:
    from models.tb_cnn import TBCNN
except ImportError:
    # Fallback: define a simple CNN for TB (32x32 input)
    import torch.nn as nn
    class TBCNN(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4,4))
            )
            self.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

def analyze_update_norms(num_clients=10, epochs=2, batches=40, dataset='skin'):
    print("="*70)
    print(f"🔬 UPDATE NORM ANALYSIS FOR {dataset.upper()} (CNN)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Select dataset and model
    if dataset == 'skin':
        client_loaders, _ = get_skin_cancer_dataloaders(num_clients=num_clients, batch_size=64)
        model_class = MediumCNN
        num_classes = 7
    elif dataset == 'chest':
        client_loaders, _ = get_chest_xray_dataloaders(num_clients=num_clients, batch_size=64)
        model_class = ChestCNN
        num_classes = 2
    elif dataset == 'tb_uganda':
        client_loaders, _ = get_tb_uganda_dataloaders(num_clients=num_clients, batch_size=32)
        model_class = TBCNN   # now defined
        num_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    update_norms = []

    for client_idx, loader in enumerate(client_loaders):
        print(f"\n📊 Client {client_idx+1}")
        model = model_class(num_classes=num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx >= batches:
                    break
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        final_weights = model.state_dict()
        squared_sum = 0.0
        for key in initial_weights:
            diff = final_weights[key] - initial_weights[key]
            squared_sum += torch.sum(diff ** 2).item()
        norm = np.sqrt(squared_sum)
        update_norms.append(norm)
        print(f"  Update L2 norm = {norm:.2f}")

    update_norms = np.array(update_norms)
    print("\n" + "="*70)
    print("📈 UPDATE NORM STATISTICS")
    print("="*70)
    print(f"Mean:        {update_norms.mean():.2f}")
    print(f"Median:      {np.median(update_norms):.2f}")
    print(f"Std Dev:     {update_norms.std():.2f}")
    print(f"Min:         {update_norms.min():.2f}")
    print(f"Max:         {update_norms.max():.2f}")
    print(f"25th %ile:   {np.percentile(update_norms, 25):.2f}")
    print(f"75th %ile:   {np.percentile(update_norms, 75):.2f}")
    print(f"95th %ile:   {np.percentile(update_norms, 95):.2f}")

    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    p50 = np.percentile(update_norms, 50)
    p75 = np.percentile(update_norms, 75)
    p95 = np.percentile(update_norms, 95)

    print(f"Conservative (50th %ile): clip_norm = {p50:.1f}")
    print(f"Moderate     (75th %ile): clip_norm = {p75:.1f}")
    print(f"Aggressive   (95th %ile): clip_norm = {p95:.1f}")

    print("\n🎯 SUGGESTED clip_norm:")
    suggested = p75
    print(f"   clip_norm = {suggested:.1f}")
    print(f"   (Clips ~25% of updates, allows 75% to pass unchanged)")

    return {
        'mean': update_norms.mean(),
        'median': np.median(update_norms),
        'std': update_norms.std(),
        'p50': p50,
        'p75': p75,
        'p95': p95,
        'suggested': suggested
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['skin', 'chest', 'tb_uganda'], default='skin',
                        help='Dataset to analyze (skin=HAM10000, chest=pneumonia, tb_uganda=TB Uganda)')
    parser.add_argument('--clients', type=int, default=10,
                        help='Number of simulated clients (default: 10)')
    args = parser.parse_args()
    stats = analyze_update_norms(num_clients=args.clients, dataset=args.dataset)
    print(f"\n💾 Use clip_norm = {stats['suggested']:.1f} for {args.dataset} dataset")