# experiments/run_tradeoff_analysis.py - UPDATED FOR SKIN CANCER
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import time
import copy
import matplotlib.pyplot as plt
from utils.data_loader import get_mnist_dataloaders

# Use the updated MNISTCNN that handles 3 channels and 7 classes
from models.mnist_cnn import MNISTCNN

from core.federated_learning import FederatedLearningBase
from core.differential_privacy import ErrorCorrectedDP

# Create results directory
os.makedirs('results', exist_ok=True)

class PrivacyTradeoffAnalyzer:
    """Analyze privacy-utility tradeoff - UPDATED FOR SKIN CANCER"""
    
    def __init__(self, num_clients=3):
        self.num_clients = num_clients
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        self.client_loaders, self.test_loader = get_mnist_dataloaders(num_clients=num_clients)
    
    def test_epsilon_range(self, epsilon_values, rounds=5, epochs_per_round=2):
        """Test different epsilon values"""
        results = {'basic_dp': [], 'ec_dp': []}
        
        for epsilon in epsilon_values:
            print(f"\n--- Testing ε={epsilon} ---")
            
            # Test Basic DP-FL
            print("  Basic DP-FL...")
            basic_acc = self._test_basic_dp(epsilon, rounds, epochs_per_round)
            results['basic_dp'].append((epsilon, basic_acc))
            
            # Test Error-Corrected DP-FL
            print("  Error-Corrected DP-FL...")
            ec_acc = self._test_ec_dp(epsilon, rounds, epochs_per_round)
            results['ec_dp'].append((epsilon, ec_acc))
            
            improvement = ec_acc - basic_acc
            print(f"  Results: Basic DP={basic_acc:.2f}%, EC-DP={ec_acc:.2f}%, Improvement={improvement:.2f}%")
        
        return results
    
    def _test_basic_dp(self, epsilon, rounds, epochs_per_round):
        """Test basic DP-FL"""
        class BasicDPFL(FederatedLearningBase):
            def __init__(self, num_clients, model_class, device, epsilon):
                super().__init__(num_clients, model_class, device)
                self.dp = ErrorCorrectedDP(epsilon=epsilon)
            
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
        
        fl = BasicDPFL(self.num_clients, MNISTCNN, self.device, epsilon)
        
        for round in range(rounds):
            fl.train_round(self.client_loaders, epochs=epochs_per_round)
            accuracy = fl.test_accuracy(self.test_loader)
            print(f"    Round {round+1}: {accuracy:.2f}%")
        
        return accuracy
    
    def _test_ec_dp(self, epsilon, rounds, epochs_per_round):
        """Test error-corrected DP-FL"""
        class ECDPFL(FederatedLearningBase):
            def __init__(self, num_clients, model_class, device, epsilon):
                super().__init__(num_clients, model_class, device)
                self.dp = ErrorCorrectedDP(epsilon=epsilon)
            
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
        
        fl = ECDPFL(self.num_clients, MNISTCNN, self.device, epsilon)
        
        for round in range(rounds):
            fl.train_round(self.client_loaders, epochs=epochs_per_round)
            accuracy = fl.test_accuracy(self.test_loader)
            print(f"    Round {round+1}: {accuracy:.2f}%")
        
        return accuracy
    
    def plot_tradeoff(self, results, save_path='results/skin_cancer_privacy_tradeoff.png'):
        """Plot privacy-utility tradeoff results"""
        epsilons_basic = [r[0] for r in results['basic_dp']]
        accuracies_basic = [r[1] for r in results['basic_dp']]
        epsilons_ec = [r[0] for r in results['ec_dp']]
        accuracies_ec = [r[1] for r in results['ec_dp']]
        
        plt.figure(figsize=(10, 6))
        
        plt.semilogx(epsilons_basic, accuracies_basic, 'ro-', label='Basic DP-FL', linewidth=3, markersize=8)
        plt.semilogx(epsilons_ec, accuracies_ec, 'bo-', label='EC-DP-FL', linewidth=3, markersize=8)
        
        plt.xlabel('Privacy Budget (ε) - Lower = More Private', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Skin Cancer: Privacy-Utility Tradeoff with Error Correction', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add improvement annotations
        for i, (eps, basic_acc, ec_acc) in enumerate(zip(epsilons_basic, accuracies_basic, accuracies_ec)):
            improvement = ec_acc - basic_acc
            if improvement > 0:
                plt.annotate(f'+{improvement:.1f}%', 
                            xy=(eps, ec_acc), 
                            xytext=(10, 10), 
                            textcoords='offset points',
                            fontsize=10,
                            color='green',
                            fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run privacy-utility tradeoff analysis for skin cancer"""
    print("🔍 Running Privacy-Utility Tradeoff Analysis - SKIN CANCER DATASET")
    print("📊 Dataset: HAM10000 (7 skin cancer types)")
    
    analyzer = PrivacyTradeoffAnalyzer(num_clients=3)
    
    # Test a range of epsilon values
    epsilon_values = [2.0, 1.0, 0.5, 0.3, 0.2, 0.1]
    results = analyzer.test_epsilon_range(epsilon_values, rounds=5, epochs_per_round=2)
    
    # Print results
    print("\n" + "="*60)
    print("📈 SKIN CANCER PRIVACY-UTILITY TRADEOFF RESULTS")
    print("="*60)
    print(f"{'ε':>8} {'Basic DP':>10} {'EC-DP':>10} {'Improvement':>12}")
    print("-"*60)
    
    total_improvement = 0
    count_improvements = 0
    
    for (eps_basic, acc_basic), (eps_ec, acc_ec) in zip(results['basic_dp'], results['ec_dp']):
        improvement = acc_ec - acc_basic
        print(f"{eps_basic:>8.1f} {acc_basic:>9.2f}% {acc_ec:>9.2f}% {improvement:>11.2f}%")
        
        if improvement > 0:
            total_improvement += improvement
            count_improvements += 1
    
    # Calculate average improvement
    if count_improvements > 0:
        avg_improvement = total_improvement / count_improvements
        print(f"\n📊 Average Improvement: {avg_improvement:.2f}% across {count_improvements} privacy levels")
    
    # Plot results
    analyzer.plot_tradeoff(results)
    print(f"\n✅ Plot saved to 'results/skin_cancer_privacy_tradeoff.png'")

if __name__ == "__main__":
    main()