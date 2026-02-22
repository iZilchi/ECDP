# utils/analyze_gradients.py
"""
Analyze actual gradient magnitudes in HAM10000 training
This will help us calibrate clip_norm properly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from utils.data_loader import get_mnist_dataloaders
from models.mnist_cnn import MNISTCNN

def analyze_gradient_norms():
    """Analyze actual gradient norms during HAM10000 training"""
    
    print("="*70)
    print("🔬 GRADIENT ANALYSIS FOR HAM10000")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders, _ = get_mnist_dataloaders(num_clients=3, batch_size=64)
    
    model = MNISTCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    gradient_norms = []
    
    print("\n📊 Training 1 epoch to measure gradient norms...\n")
    
    model.train()
    for batch_idx, (data, target) in enumerate(client_loaders[0]):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Calculate gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        gradient_norms.append(total_norm)
        
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx:3d}: Gradient Norm = {total_norm:8.2f}")
        
        if batch_idx >= 40:  # Sample 40 batches
            break
    
    # Statistics
    gradient_norms = np.array(gradient_norms)
    
    print("\n" + "="*70)
    print("📈 GRADIENT STATISTICS")
    print("="*70)
    print(f"Mean:        {gradient_norms.mean():.2f}")
    print(f"Median:      {np.median(gradient_norms):.2f}")
    print(f"Std Dev:     {gradient_norms.std():.2f}")
    print(f"Min:         {gradient_norms.min():.2f}")
    print(f"Max:         {gradient_norms.max():.2f}")
    print(f"25th %ile:   {np.percentile(gradient_norms, 25):.2f}")
    print(f"75th %ile:   {np.percentile(gradient_norms, 75):.2f}")
    print(f"95th %ile:   {np.percentile(gradient_norms, 95):.2f}")
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    # Recommend clipping based on percentiles
    p50 = np.percentile(gradient_norms, 50)
    p75 = np.percentile(gradient_norms, 75)
    p95 = np.percentile(gradient_norms, 95)
    
    print(f"Conservative (50th %ile): clip_norm = {p50:.1f}")
    print(f"Moderate (75th %ile):     clip_norm = {p75:.1f}")
    print(f"Aggressive (95th %ile):   clip_norm = {p95:.1f}")
    
    print("\n🎯 SUGGESTED clip_norm for ε=0.5:")
    suggested = p75
    print(f"   clip_norm = {suggested:.1f}")
    print(f"   (Clips ~25% of gradients, allows 75% to flow normally)")
    
    print("\n📊 Expected noise scale at ε=0.5:")
    epsilon = 0.5
    delta = 1e-5
    noise_scale = suggested * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    print(f"   σ = {noise_scale:.2f}")
    print(f"   Signal-to-Noise Ratio ≈ {(suggested**2) / (noise_scale**2):.4f}")
    
    print("\n" + "="*70)
    
    return {
        'mean': gradient_norms.mean(),
        'median': np.median(gradient_norms),
        'std': gradient_norms.std(),
        'p50': p50,
        'p75': p75,
        'p95': p95,
        'suggested': suggested
    }

if __name__ == "__main__":
    stats = analyze_gradients()
    
    print("\n💾 Gradient statistics saved!")
    print(f"📝 Use clip_norm={stats['suggested']:.1f} in your experiments")