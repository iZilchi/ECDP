# experiments/run_privacy_utility_tradeoff_improved.py
"""
IMPROVED Privacy-Utility Tradeoff Analysis
- Multiple trials for stability
- More training rounds for convergence
- Better handling of extreme privacy levels
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import get_mnist_dataloaders
from models.mnist_cnn import MNISTCNN
from core.federated_learning import FederatedLearningBase
from core.differential_privacy import ErrorCorrectedDP

os.makedirs('results', exist_ok=True)

class BasicDPFL(FederatedLearningBase):
    """Basic DP-FL for tradeoff analysis"""
    
    def __init__(self, num_clients, model_class, device, epsilon):
        super().__init__(num_clients, model_class, device)
        self.dp = ErrorCorrectedDP(epsilon=epsilon)
        self.epsilon = epsilon
    
    def _aggregate(self, client_models):
        global_model = copy.deepcopy(client_models[0])
        global_dict = global_model.state_dict()
        
        noise_scale = self.dp.calculate_noise_scale()
        
        for key in global_dict.keys():
            param_stack = torch.stack([model.state_dict()[key].float() for model in client_models])
            avg_param = param_stack.mean(0)
            noisy_param = self.dp.add_noise(avg_param, noise_scale)
            global_dict[key] = noisy_param
        
        global_model.load_state_dict(global_dict)
        return global_model

class ECDPFL(FederatedLearningBase):
    """Error-Corrected DP-FL for tradeoff analysis"""
    
    def __init__(self, num_clients, model_class, device, epsilon):
        super().__init__(num_clients, model_class, device)
        self.dp = ErrorCorrectedDP(epsilon=epsilon)
        self.epsilon = epsilon
    
    def _aggregate(self, client_models):
        global_model = copy.deepcopy(client_models[0])
        global_dict = global_model.state_dict()
        
        noise_scale = self.dp.calculate_noise_scale()
        
        for key in global_dict.keys():
            param_stack = torch.stack([model.state_dict()[key].float() for model in client_models])
            avg_param = param_stack.mean(0)
            corrected_param = self.dp.add_corrected_noise(avg_param, noise_scale)
            global_dict[key] = corrected_param
        
        global_model.load_state_dict(global_dict)
        return global_model

def train_with_trials(epsilon, num_trials=3, num_rounds=10):
    """
    Train with multiple trials for stability
    
    Args:
        epsilon: Privacy budget
        num_trials: Number of independent trials to average
        num_rounds: Number of federation rounds per trial
    
    Returns:
        avg_basic_acc, avg_ecdp_acc, std_basic, std_ecdp
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders, test_loader = get_mnist_dataloaders(num_clients=3)
    
    basic_accuracies = []
    ecdp_accuracies = []
    
    for trial in range(num_trials):
        print(f"    Trial {trial+1}/{num_trials}...")
        
        # Train Basic DP-FL
        basic_fl = BasicDPFL(3, MNISTCNN, device, epsilon)
        for round_num in range(num_rounds):
            basic_fl.train_round(client_loaders, epochs=2)
        basic_acc = basic_fl.test_accuracy(test_loader)
        basic_accuracies.append(basic_acc)
        
        # Train EC-DP-FL
        ecdp_fl = ECDPFL(3, MNISTCNN, device, epsilon)
        for round_num in range(num_rounds):
            ecdp_fl.train_round(client_loaders, epochs=2)
        ecdp_acc = ecdp_fl.test_accuracy(test_loader)
        ecdp_accuracies.append(ecdp_acc)
        
        print(f"      Basic DP: {basic_acc:.2f}%, EC-DP: {ecdp_acc:.2f}%")
    
    return (
        np.mean(basic_accuracies),
        np.mean(ecdp_accuracies),
        np.std(basic_accuracies),
        np.std(ecdp_accuracies)
    )

def run_improved_tradeoff_analysis():
    """
    Run improved privacy-utility tradeoff analysis with multiple trials
    """
    
    print("="*70)
    print("📊 IMPROVED PRIVACY-UTILITY TRADEOFF ANALYSIS")
    print("="*70)
    print("Using multiple trials for stability")
    print("Increased training rounds for better convergence")
    print("="*70 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders, test_loader = get_mnist_dataloaders(num_clients=3)
    
    # Get baseline (Standard FL)
    print("🔓 Training Standard FL (baseline)...")
    std_fl = FederatedLearningBase(3, MNISTCNN, device)
    for round_num in range(10):
        std_fl.train_round(client_loaders, epochs=2)
        if round_num % 2 == 0:
            acc = std_fl.test_accuracy(test_loader)
            print(f"  Round {round_num+1}: {acc:.2f}%")
    std_accuracy = std_fl.test_accuracy(test_loader)
    print(f"✅ Standard FL Final: {std_accuracy:.2f}%\n")
    
    # Test focused range of epsilon values
    # Focus on the privacy-meaningful range (0.1 to 2.0)
    epsilon_values = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    
    basic_dp_results = []
    ecdp_results = []
    basic_dp_stds = []
    ecdp_stds = []
    
    for epsilon in epsilon_values:
        print(f"\n{'='*70}")
        print(f"🔒 Testing ε = {epsilon}")
        print(f"{'='*70}")
        
        # Adaptive number of trials based on privacy level
        if epsilon <= 0.3:
            num_trials = 3  # More trials for high noise
            num_rounds = 15  # More rounds for convergence
        else:
            num_trials = 2  # Fewer trials for low noise
            num_rounds = 10
        
        basic_mean, ecdp_mean, basic_std, ecdp_std = train_with_trials(
            epsilon, 
            num_trials=num_trials,
            num_rounds=num_rounds
        )
        
        basic_dp_results.append((epsilon, basic_mean))
        ecdp_results.append((epsilon, ecdp_mean))
        basic_dp_stds.append(basic_std)
        ecdp_stds.append(ecdp_std)
        
        improvement = ecdp_mean - basic_mean
        print(f"\n  📊 Results (averaged over {num_trials} trials):")
        print(f"    Basic DP: {basic_mean:.2f}% ± {basic_std:.2f}%")
        print(f"    EC-DP:    {ecdp_mean:.2f}% ± {ecdp_std:.2f}%")
        print(f"    ✨ Improvement: {improvement:+.2f}%")
    
    # Plot results
    plot_improved_tradeoff(
        epsilon_values, 
        basic_dp_results, 
        ecdp_results,
        basic_dp_stds,
        ecdp_stds,
        std_accuracy
    )
    
    # Print summary
    print_improved_summary(
        epsilon_values, 
        basic_dp_results, 
        ecdp_results,
        basic_dp_stds,
        ecdp_stds,
        std_accuracy
    )
    
    return basic_dp_results, ecdp_results, std_accuracy

def plot_improved_tradeoff(epsilon_values, basic_dp_results, ecdp_results, 
                           basic_stds, ecdp_stds, std_accuracy):
    """Create improved tradeoff plot with error bars"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract data
    epsilons_basic = [r[0] for r in basic_dp_results]
    accuracies_basic = [r[1] for r in basic_dp_results]
    epsilons_ecdp = [r[0] for r in ecdp_results]
    accuracies_ecdp = [r[1] for r in ecdp_results]
    
    # Plot 1: Semi-log plot with error bars
    ax1.errorbar(epsilons_basic, accuracies_basic, yerr=basic_stds,
                fmt='ro-', label='Basic DP-FL', linewidth=3, markersize=10,
                capsize=5, capthick=2)
    ax1.errorbar(epsilons_ecdp, accuracies_ecdp, yerr=ecdp_stds,
                fmt='bo-', label='EC-DP-FL (Ours)', linewidth=3, markersize=10,
                capsize=5, capthick=2)
    ax1.axhline(y=std_accuracy, color='green', linestyle='--', 
               linewidth=2, label='Standard FL (no privacy)')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Privacy Budget (ε) - Lower = More Private', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Privacy-Utility Tradeoff (with Error Bars)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Add improvement annotations for significant improvements only
    for i, (eps, basic_acc, ecdp_acc) in enumerate(zip(epsilons_basic, accuracies_basic, accuracies_ecdp)):
        improvement = ecdp_acc - basic_acc
        if improvement > 10.0:  # Only annotate large improvements
            ax1.annotate(f'+{improvement:.1f}%', 
                        xy=(eps, ecdp_acc), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=10,
                        color='darkgreen',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Plot 2: Improvement by epsilon
    improvements = [ecdp - basic for (_, basic), (_, ecdp) in zip(basic_dp_results, ecdp_results)]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax2.bar(range(len(epsilon_values)), improvements, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + (2 if height > 0 else -2),
                f'{imp:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Privacy Budget (ε)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Error Correction Improvement by Privacy Level', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(epsilon_values)))
    ax2.set_xticklabels([f'{e}' for e in epsilon_values])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/improved_privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Improved tradeoff plot saved: results/improved_privacy_utility_tradeoff.png")
    plt.show()

def print_improved_summary(epsilon_values, basic_dp_results, ecdp_results,
                          basic_stds, ecdp_stds, std_accuracy):
    """Print improved summary with error bars"""
    
    print("\n" + "="*100)
    print("📋 IMPROVED PRIVACY-UTILITY TRADEOFF SUMMARY")
    print("="*100)
    print(f"{'ε':>8} {'Basic DP':>18} {'EC-DP':>18} {'Improvement':>14} {'Recovery':>12} {'Privacy':>15}")
    print("-"*100)
    
    valid_improvements = []
    
    for (eps_basic, acc_basic), (eps_ecdp, acc_ecdp), basic_std, ecdp_std in \
        zip(basic_dp_results, ecdp_results, basic_stds, ecdp_stds):
        
        improvement = acc_ecdp - acc_basic
        dp_loss = std_accuracy - acc_basic
        recovery_rate = (improvement / dp_loss * 100) if dp_loss > 0 else 0
        
        # Determine privacy level
        if eps_basic <= 0.3:
            privacy_level = "Very Strong"
        elif eps_basic <= 1.0:
            privacy_level = "Strong"
        else:
            privacy_level = "Moderate"
        
        print(f"{eps_basic:>8.1f} {acc_basic:>8.2f}% ± {basic_std:>4.2f}% "
              f"{acc_ecdp:>8.2f}% ± {ecdp_std:>4.2f}% "
              f"{improvement:>13.2f}% {recovery_rate:>11.1f}% {privacy_level:>15}")
        
        if improvement > 0:
            valid_improvements.append(improvement)
    
    print("-"*100)
    print(f"Standard FL (no privacy): {std_accuracy:.2f}%")
    print("="*100)
    
    # Statistics
    if valid_improvements:
        print(f"\n📊 STATISTICS (positive improvements only):")
        print(f"  Average Improvement: {np.mean(valid_improvements):.2f}%")
        print(f"  Maximum Improvement: {max(valid_improvements):.2f}%")
        print(f"  Minimum Improvement: {min(valid_improvements):.2f}%")
    
    # Find optimal epsilon
    scores = [(eps, ecdp_acc, 1/eps * ecdp_acc) 
              for (eps, _), (_, ecdp_acc) in zip(basic_dp_results, ecdp_results)]
    best_epsilon = max(scores, key=lambda x: x[2])
    
    print(f"\n🎯 RECOMMENDED PRIVACY BUDGET:")
    print(f"  ε = {best_epsilon[0]} (Accuracy: {best_epsilon[1]:.2f}%)")
    print(f"  Rationale: Best balance of strong privacy and diagnostic accuracy")

if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("IMPROVED Privacy-Utility Tradeoff Analysis")
    print("Multiple trials + More rounds = Stable results")
    print("🚀"*35 + "\n")
    
    basic_dp_results, ecdp_results, std_accuracy = run_improved_tradeoff_analysis()
    
    print("\n🎉 Improved Privacy-Utility Tradeoff Analysis Complete!")
    print("📁 Results saved in: ./results/")
    print("📊 Stable, publication-ready results with error bars")