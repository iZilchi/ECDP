"""
Analyze model update norms after local training.
This helps calibrate the clipping norm (C) for differential privacy.
Supports both HAM10000 (skin cancer) and Chest X-ray datasets.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from utils.data_loader import get_skin_cancer_dataloaders
from utils.chest_xray_loader import get_chest_xray_dataloaders
from models.medium_cnn import MediumCNN
from models.chest_xray_cnn import ChestXRayCNN

def get_model_constructor(dataset, use_tiny=False):
    """Return model class for the dataset."""
    if dataset == 'skin_cancer':
        return MediumCNN
    elif dataset == 'chest_xray':
        return ChestXRayCNN
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_dataloaders(dataset, num_clients, batch_size, alpha, seed):
    """Return (client_loaders, test_loader) for chosen dataset."""
    if dataset == 'skin_cancer':
        return get_skin_cancer_dataloaders(
            num_clients=num_clients,
            batch_size=batch_size,
            alpha=alpha,
            seed=seed
        )
    elif dataset == 'chest_xray':
        return get_chest_xray_dataloaders(
            num_clients=num_clients,
            batch_size=batch_size,
            alpha=alpha,
            seed=seed,
            data_root='./data/chest_xray'
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def analyze_update_norms(dataset='skin_cancer', num_clients=3, epochs=2, batches=40,
                         batch_size=64, alpha=None, seed=42, use_tiny=False):
    """Analyze gradient norms for a given dataset."""
    print("="*70)
    print(f"🔬 UPDATE NORM ANALYSIS FOR {dataset.upper()} ({'TinyCNN' if use_tiny else 'Default CNN'})")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_constructor = get_model_constructor(dataset, use_tiny)
    client_loaders, _ = get_dataloaders(dataset, num_clients, batch_size, alpha, seed)

    update_norms = []

    for client_idx, loader in enumerate(client_loaders):
        print(f"\n📊 Client {client_idx+1}")
        model = model_constructor().to(device)
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

    print(f"Conservative (50th %ile): clip_norm ≈ {p50:.1f}")
    print(f"Moderate     (75th %ile): clip_norm ≈ {p75:.1f}")
    print(f"Aggressive   (95th %ile): clip_norm ≈ {p95:.1f}")

    candidates = [0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0]
    closest_candidate = min(candidates, key=lambda x: abs(x - p75))
    print(f"\n🎯 SUGGESTED clip_norm (from candidates {candidates}):")
    print(f"   clip_norm = {closest_candidate}")
    print(f"   (Based on 75th percentile, which clips ~25% of updates)")
    
    return {
        'mean': update_norms.mean(),
        'median': np.median(update_norms),
        'std': update_norms.std(),
        'p50': p50,
        'p75': p75,
        'p95': p95,
        'suggested': closest_candidate
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze gradient norms for clipping norm selection.')
    parser.add_argument('--dataset', choices=['skin_cancer', 'chest_xray'], default='skin_cancer',
                        help='Dataset to analyze')
    parser.add_argument('--num_clients', type=int, default=3,
                        help='Number of simulated clients')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of local epochs per client')
    parser.add_argument('--batches', type=int, default=40,
                        help='Number of batches to process per client')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--iid', action='store_true',
                        help='Use IID data distribution (default: Non-IID α=0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_tiny', action='store_true',
                        help='Use TinyCNN instead of dataset-specific model (for quick tests)')
    args = parser.parse_args()

    alpha = None if args.iid else 0.5
    stats = analyze_update_norms(
        dataset=args.dataset,
        num_clients=args.num_clients,
        epochs=args.epochs,
        batches=args.batches,
        batch_size=args.batch_size,
        alpha=alpha,
        seed=args.seed,
        use_tiny=args.use_tiny
    )
    print(f"\n💾 Recommended clip_norm for {args.dataset} = {stats['suggested']}")