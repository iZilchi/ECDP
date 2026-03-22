import argparse
import torch
import sys
import os
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import get_skin_cancer_dataloaders
from utils.chest_xray_loader import get_chest_xray_dataloaders
from models.medium_cnn import MediumCNN
from models.chest_xray_cnn import ChestXRayCNN

from core.federated_learning import FederatedLearningBase as StandardFL
from core.dpfl import BasicDPFL, ECDPFL
from utils.metrics import compare_methods_comprehensive
import matplotlib.pyplot as plt

BATCH_SIZE = 32

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset_components(dataset_name, num_clients=3, batch_size=BATCH_SIZE, alpha=None, seed=42):
    if dataset_name == 'skin':
        client_loaders, test_loader = get_skin_cancer_dataloaders(
            num_clients=num_clients, batch_size=batch_size, alpha=alpha, seed=seed
        )
        model_class = lambda: MediumCNN(num_classes=7)
        num_classes = 7
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    else:  # chest
        client_loaders, test_loader = get_chest_xray_dataloaders(
            num_clients=num_clients, batch_size=batch_size, alpha=alpha, seed=seed
        )
        model_class = lambda: ChestXRayCNN(num_classes=2)
        num_classes = 2
        class_names = ['NORMAL', 'PNEUMONIA']
    return client_loaders, test_loader, model_class, num_classes, class_names

def run_comparison(per_round_epsilon=None, target_epsilon=None, clip_norm=3.5, num_rounds=20,
                   device='cpu', c=1.5, alpha_corr=0.6, seed=42, plot=True, dataset='skin', alpha_data=None):
    print(f"\n{'='*60}")
    if per_round_epsilon is not None:
        mode = "per‑round ε"
        eps = per_round_epsilon
    else:
        mode = "total ε"
        eps = target_epsilon
    print(f"COMPARISON: {mode}={eps} over {num_rounds} rounds, clip_norm={clip_norm}, c={c}, α={alpha_corr}, seed={seed}, dataset={dataset}, alpha_data={alpha_data}")
    print('='*60)

    set_seed(seed)

    client_loaders, test_loader, model_class, num_classes, class_names = get_dataset_components(
        dataset, alpha=alpha_data, seed=seed
    )

    std_fl = StandardFL(3, model_class, device)

    if per_round_epsilon is not None:
        dp_fl = BasicDPFL(3, model_class, device,
                          epsilon=per_round_epsilon,
                          clip_norm=clip_norm)
        ecdp_fl = ECDPFL(3, model_class, device,
                         epsilon=per_round_epsilon,
                         clip_norm=clip_norm,
                         c=c, alpha=alpha_corr)
    else:
        dp_fl = BasicDPFL(3, model_class, device,
                          epsilon=None, target_epsilon=target_epsilon, max_rounds=num_rounds,
                          clip_norm=clip_norm)
        ecdp_fl = ECDPFL(3, model_class, device,
                         epsilon=None, target_epsilon=target_epsilon, max_rounds=num_rounds,
                         clip_norm=clip_norm,
                         c=c, alpha=alpha_corr)

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

    metrics = compare_methods_comprehensive(std_fl, dp_fl, ecdp_fl, test_loader, device,
                                            num_classes=num_classes, class_names=class_names)

    if plot:
        plt.figure(figsize=(10,6))
        for name, acc in histories.items():
            plt.plot(range(1, len(acc)+1), acc, marker='o', label=name)
        plt.xlabel('Federation Round')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Convergence ({mode}={eps}) - {dataset} (alpha_data={alpha_data})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/convergence_{mode}_{eps}_{dataset}_alpha{alpha_data}_seed{seed}.png', dpi=150)
        plt.show()

    return metrics, histories, test_loader

def run_tradeoff(epsilon_values, clip_norm, num_rounds=20, device='cpu', base_seed=42, mode='per_round', dataset='skin', alpha_data=None):
    print("\n" + "="*70)
    print(f"PRIVACY‑UTILITY TRADEOFF ANALYSIS ({mode} ε) - {dataset} (alpha_data={alpha_data})")
    print("="*70)

    basic_means, ecdp_means = [], []
    basic_stds, ecdp_stds = [], []   # single run, std=0

    for eps in epsilon_values:
        print(f"\n--- {mode} ε = {eps} ---")
        set_seed(base_seed)
        client_loaders, test_loader, model_class, _, _ = get_dataset_components(
            dataset, alpha=alpha_data, seed=base_seed
        )

        # Heuristic correction parameters (no warmup)
        if eps <= 0.5:
            c, alpha_corr = 1.5, 0.6
        else:
            c, alpha_corr = 2.5, 0.8

        if mode == 'per_round':
            dp = BasicDPFL(3, model_class, device,
                           epsilon=eps,
                           clip_norm=clip_norm)
            ec = ECDPFL(3, model_class, device,
                        epsilon=eps,
                        clip_norm=clip_norm,
                        c=c, alpha=alpha_corr)
        else:
            dp = BasicDPFL(3, model_class, device,
                           epsilon=None, target_epsilon=eps, max_rounds=num_rounds,
                           clip_norm=clip_norm)
            ec = ECDPFL(3, model_class, device,
                        epsilon=None, target_epsilon=eps, max_rounds=num_rounds,
                        clip_norm=clip_norm,
                        c=c, alpha=alpha_corr)

        for r in range(num_rounds):
            dp.train_round(client_loaders, epochs=2)
            ec.train_round(client_loaders, epochs=2)

        basic_acc = dp.test_accuracy(test_loader)
        ecdp_acc = ec.test_accuracy(test_loader)

        basic_means.append(basic_acc)
        ecdp_means.append(ecdp_acc)
        basic_stds.append(0.0)
        ecdp_stds.append(0.0)

        print(f"  Basic DP: {basic_acc:.2f}%")
        print(f"  EC-DP:    {ecdp_acc:.2f}%")

    # Plot tradeoff
    plt.figure(figsize=(10,6))
    plt.errorbar(epsilon_values, basic_means, yerr=basic_stds, marker='o', label='Basic DP-FL', capsize=5)
    plt.errorbar(epsilon_values, ecdp_means, yerr=ecdp_stds, marker='s', label='EC-DP-FL', capsize=5)
    # Standard FL accuracy (run once)
    set_seed(base_seed)
    client_loaders, test_loader, model_class, _, _ = get_dataset_components(
        dataset, alpha=alpha_data, seed=base_seed
    )
    std_fl = StandardFL(3, model_class, device)
    for r in range(num_rounds):
        std_fl.train_round(client_loaders, epochs=2)
    std_acc = std_fl.test_accuracy(test_loader)
    plt.axhline(y=std_acc, color='g', linestyle='--', label='Standard FL')
    plt.xscale('log')
    plt.xlabel(f'Privacy budget ε ({mode})')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Privacy‑Utility Tradeoff ({mode} ε) - {dataset} (alpha_data={alpha_data})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/tradeoff_{mode}_{dataset}_alpha{alpha_data}.png', dpi=150)
    plt.show()

def run_tune(per_round_epsilon=None, target_epsilon=None, clip_norm=3.5, num_rounds=20,
             device='cpu', seed=42, dataset='skin', alpha_data=None,
             c_values=[1.5, 2.0, 2.5], alpha_corr_values=[0.6, 0.7, 0.8]):
    """
    Grid search over c and alpha_corr values. For each combination,
    runs a full comparison (without plotting) and records final EC-DP-FL accuracy.
    """
    if per_round_epsilon is not None:
        mode = "per‑round ε"
        eps = per_round_epsilon
    else:
        mode = "total ε"
        eps = target_epsilon
    print("\n" + "="*70)
    print(f"TUNING CORRECTION PARAMETERS ({mode}={eps})")
    print(f"c_values = {c_values}")
    print(f"alpha_corr_values = {alpha_corr_values}")
    print("="*70)

    best_acc = -1
    best_params = None
    results = []

    for c in c_values:
        for alpha_corr in alpha_corr_values:
            print(f"\n--- Testing c={c}, α={alpha_corr} ---")
            # Run comparison with plot=False to avoid showing plots
            _, histories, _ = run_comparison(
                per_round_epsilon, target_epsilon, clip_norm, num_rounds,
                device, c, alpha_corr, seed=seed, plot=False,
                dataset=dataset, alpha_data=alpha_data
            )
            acc = histories['EC-DP-FL'][-1]
            results.append((c, alpha_corr, acc))
            print(f"  Final EC-DP-FL accuracy: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_params = {'c': c, 'alpha': alpha_corr}

    print("\n" + "="*70)
    print("TUNING RESULTS")
    print("="*70)
    for c, alpha_corr, acc in results:
        print(f"c={c}, α={alpha_corr}: {acc:.2f}%")
    print(f"\nBest params: c={best_params['c']}, α={best_params['alpha']} with accuracy {best_acc:.2f}%")
    print("="*70)

    return best_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['comparison', 'tradeoff', 'tune'], default='comparison')
    parser.add_argument('--dataset', choices=['skin', 'chest'], default='skin',
                        help='Choose dataset: skin (HAM10000) or chest (pneumonia)')
    # Privacy budget: either per-round or total
    parser.add_argument('--per_round_epsilon', type=float, default=None,
                        help='Per‑round privacy budget (if using per‑round interpretation)')
    parser.add_argument('--target_epsilon', type=float, default=None,
                        help='Total privacy budget over all rounds (if using total interpretation)')
    parser.add_argument('--clip_norm', type=float, default=3.5,
                        help='Clipping norm (suggested from analyze_gradients.py)')
    parser.add_argument('--rounds', type=int, default=20,
                        help='Number of federation rounds')
    parser.add_argument('--device', default=None,
                        help='Device to use: cuda, cpu, or auto (default)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # Correction parameters (used in comparison and tune)
    parser.add_argument('--c', type=float, default=1.5, help='Correction bound parameter (for comparison mode)')
    parser.add_argument('--alpha_corr', type=float, default=0.6, help='Smoothing coefficient (for comparison mode)')
    # Data heterogeneity
    parser.add_argument('--alpha_data', type=float, default=None,
                        help='Dirichlet concentration parameter for non-IID data. If None, IID.')
    # Tuning-specific arguments
    parser.add_argument('--c_values', nargs='+', type=float, default=[1.5, 2.0, 2.5],
                        help='List of c values to try during tuning')
    parser.add_argument('--alpha_corr_values', nargs='+', type=float, default=[0.6, 0.7, 0.8],
                        help='List of alpha values to try during tuning')

    args = parser.parse_args()

    # Device auto-detection
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    if args.mode == 'comparison':
        assert (args.per_round_epsilon is not None) ^ (args.target_epsilon is not None), \
            "Exactly one of --per_round_epsilon or --target_epsilon must be provided."
        run_comparison(args.per_round_epsilon, args.target_epsilon, args.clip_norm, args.rounds, device,
                       args.c, args.alpha_corr, seed=args.seed, dataset=args.dataset,
                       alpha_data=args.alpha_data)
    elif args.mode == 'tradeoff':
        epsilon_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        run_tradeoff(epsilon_list, args.clip_norm, args.rounds,
                     device=device, base_seed=args.seed, mode='per_round',
                     dataset=args.dataset, alpha_data=args.alpha_data)
    elif args.mode == 'tune':
        assert (args.per_round_epsilon is not None) ^ (args.target_epsilon is not None), \
            "Exactly one of --per_round_epsilon or --target_epsilon must be provided."
        run_tune(args.per_round_epsilon, args.target_epsilon, args.clip_norm, args.rounds, device,
                 seed=args.seed, dataset=args.dataset, alpha_data=args.alpha_data,
                 c_values=args.c_values, alpha_corr_values=args.alpha_corr_values)
                 