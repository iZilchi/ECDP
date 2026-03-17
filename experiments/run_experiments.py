import argparse
import torch
import sys
import os
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import get_skin_cancer_dataloaders
from models.tiny_cnn import TinyCNN
from core.federated_learning import FederatedLearningBase as StandardFL
from core.dpfl import BasicDPFL, ECDPFL
from utils.metrics import compare_methods_comprehensive
import matplotlib.pyplot as plt

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior on GPU (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_comparison(epsilon, clip_norm, num_rounds=10, device='cpu',
                   c=2.5, alpha=0.8, warm_up=5, seed=42, plot=True):
    print(f"\n{'='*60}")
    print(f"COMPARISON: ε={epsilon}, clip_norm={clip_norm}, c={c}, α={alpha}, warm_up={warm_up}, seed={seed}")
    print('='*60)

    set_seed(seed)  # <-- set seed before creating loaders

    client_loaders, test_loader = get_skin_cancer_dataloaders(num_clients=3)

    std_fl = StandardFL(3, TinyCNN, device)
    dp_fl = BasicDPFL(3, TinyCNN, device, epsilon, clip_norm=clip_norm)
    ecdp_fl = ECDPFL(3, TinyCNN, device, epsilon, clip_norm=clip_norm,
                     c=c, alpha=alpha, warm_up=warm_up)

    methods = {'Standard FL': std_fl, 'Basic DP-FL': dp_fl, 'EC-DP-FL': ecdp_fl}
    histories = {}

    for name, method in methods.items():
        print(f"\n--- Training {name} ---")
        acc_list = []
        for r in range(num_rounds):
            method.train_round(client_loaders, epochs=2)
            acc = method.test_accuracy(test_loader)
            acc_list.append(acc)
            print(f"Round {r+1}: {acc:.2f}%")
        histories[name] = acc_list

    metrics = compare_methods_comprehensive(std_fl, dp_fl, ecdp_fl, test_loader, device)

    if plot:
        plt.figure(figsize=(10,6))
        for name, acc in histories.items():
            plt.plot(range(1, len(acc)+1), acc, marker='o', label=name)
        plt.xlabel('Federation Round')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Convergence (ε={epsilon}, C={clip_norm}, c={c}, α={alpha})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/convergence_eps{epsilon}_seed{seed}.png', dpi=150)
        plt.show()

    return metrics, histories, test_loader

def tune_correction_params(epsilon, clip_norm, num_rounds=10, device='cpu',
                           c_values=[1.5, 2.0, 2.5],
                           alpha_values=[0.6, 0.7, 0.8],
                           warm_up_values=[0, 3],
                           seed=42):
    """Sensitivity analysis for correction parameters."""
    best_acc = -1
    best_params = {}
    for c in c_values:
        for alpha in alpha_values:
            for warm_up in warm_up_values:
                print(f"\n--- Testing c={c}, α={alpha}, warm_up={warm_up} ---")
                # Use a different seed for each combo? For fairness, use the same base seed.
                # We'll just pass the same seed to each run.
                _, histories, _ = run_comparison(epsilon, clip_norm, num_rounds, device,
                                                  c, alpha, warm_up, seed=seed, plot=False)
                acc = histories['EC-DP-FL'][-1]
                print(f"Final EC-DP-FL accuracy: {acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'c': c, 'alpha': alpha, 'warm_up': warm_up}
    print(f"\nBest params: {best_params} with accuracy {best_acc:.2f}%")
    return best_params

def run_tradeoff(epsilon_values, clip_norm, num_rounds=10, num_trials=3, device='cpu', base_seed=42):
    print("\n" + "="*70)
    print("PRIVACY‑UTILITY TRADEOFF ANALYSIS")
    print("="*70)

    # We'll run multiple trials with different seeds
    basic_means, ecdp_means = [], []
    basic_stds, ecdp_stds = [], []

    for eps in epsilon_values:
        print(f"\n--- ε = {eps} ---")
        basic_accs, ecdp_accs = [], []
        for trial in range(num_trials):
            seed = base_seed + trial  # different seed per trial
            print(f"  Trial {trial+1}/{num_trials} (seed={seed})")
            # For tradeoff, we might want fixed correction params; using defaults from manuscript:
            # For high privacy (small ε) use aggressive, for low privacy use mild.
            if eps <= 0.5:
                c, alpha, warm_up = 1.5, 0.6, 3
            else:
                c, alpha, warm_up = 2.5, 0.8, 0
            # Or you could use tuned params from previous runs; here we keep it simple.
            dp = BasicDPFL(3, TinyCNN, device, eps, clip_norm=clip_norm)
            ec = ECDPFL(3, TinyCNN, device, eps, clip_norm=clip_norm,
                        c=c, alpha=alpha, warm_up=warm_up)
            # Train both methods with same seed for fair comparison
            set_seed(seed)
            client_loaders, test_loader = get_skin_cancer_dataloaders(num_clients=3)
            for r in range(num_rounds):
                dp.train_round(client_loaders, epochs=2)
                ec.train_round(client_loaders, epochs=2)
            basic_accs.append(dp.test_accuracy(test_loader))
            ecdp_accs.append(ec.test_accuracy(test_loader))

        basic_means.append(np.mean(basic_accs))
        basic_stds.append(np.std(basic_accs))
        ecdp_means.append(np.mean(ecdp_accs))
        ecdp_stds.append(np.std(ecdp_accs))

        print(f"  Basic DP: {basic_means[-1]:.2f} ± {basic_stds[-1]:.2f}%")
        print(f"  EC-DP:    {ecdp_means[-1]:.2f} ± {ecdp_stds[-1]:.2f}%")

    # Plot tradeoff
    plt.figure(figsize=(10,6))
    plt.errorbar(epsilon_values, basic_means, yerr=basic_stds, marker='o', label='Basic DP-FL', capsize=5)
    plt.errorbar(epsilon_values, ecdp_means, yerr=ecdp_stds, marker='s', label='EC-DP-FL', capsize=5)
    # Standard FL accuracy (run once with a fixed seed)
    set_seed(base_seed)
    client_loaders, test_loader = get_skin_cancer_dataloaders(num_clients=3)
    std_fl = StandardFL(3, TinyCNN, device)
    for r in range(num_rounds):
        std_fl.train_round(client_loaders, epochs=2)
    std_acc = std_fl.test_accuracy(test_loader)
    plt.axhline(y=std_acc, color='g', linestyle='--', label='Standard FL')
    plt.xscale('log')
    plt.xlabel('Privacy budget ε (lower = more private)')
    plt.ylabel('Accuracy (%)')
    plt.title('Privacy‑Utility Tradeoff')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/tradeoff.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['comparison', 'tradeoff', 'tune'], default='comparison')
    parser.add_argument('--epsilon', type=float, default=20.0)
    parser.add_argument('--clip_norm', type=float, default=2.1)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # Correction parameters (used if mode=comparison)
    parser.add_argument('--c', type=float, default=2.5)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--warm_up', type=int, default=5)
    args = parser.parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.mode == 'comparison':
        run_comparison(args.epsilon, args.clip_norm, args.rounds, device,
                       args.c, args.alpha, args.warm_up, seed=args.seed)
    elif args.mode == 'tradeoff':
        epsilon_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        run_tradeoff(epsilon_list, args.clip_norm, args.rounds, num_trials=3,
                     device=device, base_seed=args.seed)
    elif args.mode == 'tune':
        tune_correction_params(args.epsilon, args.clip_norm, args.rounds, device,
                               seed=args.seed)