# experiments/run_comparison.py - UPDATED FOR SKIN CANCER
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import get_mnist_dataloaders

# Use the updated MNISTCNN that handles 3 channels and 7 classes
from models.mnist_cnn import MNISTCNN

from core.federated_learning import FederatedLearningBase
from core.differential_privacy import ErrorCorrectedDP

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

class StandardFL(FederatedLearningBase):
    """Standard FL without privacy"""
    pass

class DPFL(FederatedLearningBase):
    """FL with basic DP"""
    
    def __init__(self, num_clients, model_class, device, epsilon=1.0):
        super().__init__(num_clients, model_class, device)
        self.dp = ErrorCorrectedDP(epsilon=epsilon)
    
    def _aggregate(self, client_models):
        """Aggregate with DP"""
        global_model = copy.deepcopy(client_models[0])
        global_dict = global_model.state_dict()
        
        noise_scale = self.dp.calculate_noise_scale()
        
        for key in global_dict.keys():
            param_stack = torch.stack([model.state_dict()[key].float() for model in client_models])
            avg_param = param_stack.mean(0)
            
            # Add DP noise
            noisy_param = self.dp.add_noise(avg_param, noise_scale)
            global_dict[key] = noisy_param
        
        global_model.load_state_dict(global_dict)
        return global_model

class ECDPFL(DPFL):
    """FL with Error-Corrected DP (Our Method)"""
    
    def _aggregate(self, client_models):
        """Aggregate with error-corrected DP"""
        global_model = copy.deepcopy(client_models[0])
        global_dict = global_model.state_dict()
        
        noise_scale = self.dp.calculate_noise_scale()
        
        for key in global_dict.keys():
            param_stack = torch.stack([model.state_dict()[key].float() for model in client_models])
            avg_param = param_stack.mean(0)
            
            # Add error-corrected DP noise
            corrected_param = self.dp.add_corrected_noise(avg_param, noise_scale)
            global_dict[key] = corrected_param
        
        global_model.load_state_dict(global_dict)
        return global_model

def run_comprehensive_comparison():
    """Run main comparison experiment - UPDATED FOR SKIN CANCER"""
    print("🎯 Running Comprehensive FL Comparison - SKIN CANCER DATASET")
    print("📊 Dataset: HAM10000 (7 skin cancer types)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    client_loaders, test_loader = get_mnist_dataloaders(num_clients=3, batch_size=64)
    
    # Use privacy settings for skin cancer
    methods = {
        'Standard FL': StandardFL(3, MNISTCNN, device),
        'Basic DP-FL': DPFL(3, MNISTCNN, device, epsilon=0.5),
        'EC-DP-FL': ECDPFL(3, MNISTCNN, device, epsilon=0.5)
    }
    
    results = {}
    
    for name, method in methods.items():
        print(f"\n=== {name} ===")
        accuracies = []
        times = []
        
        for round in range(5):
            start_time = time.time()
            method.train_round(client_loaders, epochs=2)
            accuracy = method.test_accuracy(test_loader)
            round_time = time.time() - start_time
            
            accuracies.append(accuracy)
            times.append(round_time)
            print(f"Round {round+1}: {accuracy:.2f}% | {round_time:.2f}s")
            
        results[name] = {'accuracies': accuracies, 'times': times}
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print_summary(results)
    return results

def plot_results(results):
    """Plot comparison results"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy progression
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        rounds = range(1, len(data['accuracies']) + 1)
        plt.plot(rounds, data['accuracies'], 'o-', label=name, markersize=8, linewidth=2)
    
    plt.xlabel('Federation Round')
    plt.ylabel('Accuracy (%)')
    plt.title('Skin Cancer Classification Accuracy (ε=0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Time comparison
    plt.subplot(1, 2, 2)
    names = list(results.keys())
    avg_times = [np.mean(data['times']) for data in results.values()]
    
    colors = ['green', 'red', 'blue']
    bars = plt.bar(names, avg_times, color=colors, alpha=0.7)
    for bar, time_val in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Average Time per Round (s)')
    plt.title('Computational Overhead')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/skin_cancer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(results):
    """Print summary of results"""
    print("\n" + "="*60)
    print("📊 SKIN CANCER CLASSIFICATION SUMMARY (ε=0.5)")
    print("="*60)
    print(f"{'Method':<15} {'Final Accuracy':<15} {'Best Accuracy':<15} {'Avg Time':<10}")
    print("-"*60)
    
    for name, data in results.items():
        final_acc = data['accuracies'][-1] if data['accuracies'] else 0
        best_acc = max(data['accuracies']) if data['accuracies'] else 0
        avg_time = np.mean(data['times']) if data['times'] else 0
        
        print(f"{name:<15} {final_acc:<14.2f}% {best_acc:<14.2f}% {avg_time:<9.1f}s")
    
    # Calculate improvements
    if 'Standard FL' in results and 'Basic DP-FL' in results and 'EC-DP-FL' in results:
        std_final = results['Standard FL']['accuracies'][-1]
        basic_final = results['Basic DP-FL']['accuracies'][-1]
        ec_final = results['EC-DP-FL']['accuracies'][-1]
        
        dp_drop = std_final - basic_final
        ec_drop = std_final - ec_final
        improvement = ec_final - basic_final
        
        print("\n" + "="*60)
        print("🔍 PRIVACY-UTILITY ANALYSIS")
        print("="*60)
        print(f"Standard FL (no privacy): {std_final:.2f}%")
        print(f"Basic DP-FL utility loss: {dp_drop:.2f}%")
        print(f"EC-DP-FL utility loss:    {ec_drop:.2f}%")
        print(f"EC-DP-FL improvement:     +{improvement:.2f}%")
        if dp_drop > 0:
            print(f"Error correction recovers: {improvement/dp_drop*100:.1f}% of lost utility")

if __name__ == "__main__":
    run_comprehensive_comparison()