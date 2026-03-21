"""
Analyze model update norms after local training.
This helps calibrate the clipping norm (C) for differential privacy.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from utils.data_loader import get_skin_cancer_dataloaders
from models.tiny_cnn import TinyCNN

def analyze_update_norms(num_clients=3, epochs=2, batches=40):
    print("="*70)
    print("🔬 UPDATE NORM ANALYSIS FOR HAM10000 (TinyCNN)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders, _ = get_skin_cancer_dataloaders(num_clients=num_clients, batch_size=64)

    update_norms = []

    for client_idx, loader in enumerate(client_loaders):
        print(f"\n📊 Client {client_idx+1}")
        model = TinyCNN().to(device)
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
    stats = analyze_update_norms()
    print(f"\n💾 Use clip_norm = {stats['suggested']:.1f} in your experiments")