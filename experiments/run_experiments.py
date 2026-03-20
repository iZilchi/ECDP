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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset_components(dataset_name, num_clients=3, batch_size=32, alpha=None, seed=42):
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

def run_comparison(per_round_epsilon=None, target_epsilon=None, clip_norm=2.3,
                   num_rounds=20, device='cpu',
                   c=2.5, alpha=0.8, warm_up=0,
                   participation_rate=0.5, seed=42, plot=True,
                   dataset='skin', dirichlet_alpha=None, batch_size=32, local_epochs=3):
    print(f"\n{'='*60}")
    if per_round_epsilon is not None:
        mode = "per‑round ε"
        eps = per_round_epsilon
    else:
        mode = "total ε"
        eps = target_epsilon
    print(f"COMPARISON: {mode}={eps} over {num_rounds} rounds, clip_norm={clip_norm}, c={c}, α={alpha}, warm_up={warm_up}, seed={seed}, dataset={dataset}, participation={participation_rate}, alpha_dirichlet={dirichlet_alpha}, batch_size={batch_size}, local_epochs={local_epochs}")
    print('='*60)

    set_seed(seed)

    # Load data with Dirichlet and batch size
    client_loaders, test_loader, model_class, num_classes, class_names = get_dataset_components(
        dataset, num_clients=3, batch_size=batch_size, alpha=dirichlet_alpha, seed=seed)

    # Initialize methods with participation rate
    std_fl = StandardFL(3, model_class, device, participation_rate=participation_rate)

    if per_round_epsilon is not None:
        dp_fl = BasicDPFL(3, model_class, device, participation_rate=participation_rate,
                          epsilon=per_round_epsilon, clip_norm=clip_norm)
        ecdp_fl = ECDPFL(3, model_class, device, participation_rate=participation_rate,
                         epsilon=per_round_epsilon, clip_norm=clip_norm,
                         c=c, alpha=alpha, warm_up=warm_up)
    else:
        dp_fl = BasicDPFL(3, model_class, device, participation_rate=participation_rate,
                          epsilon=None, target_epsilon=target_epsilon, max_rounds=num_rounds,
                          clip_norm=clip_norm)
        ecdp_fl = ECDPFL(3, model_class, device, participation_rate=participation_rate,
                         epsilon=None, target_epsilon=target_epsilon, max_rounds=num_rounds,
                         clip_norm=clip_norm,
                         c=c, alpha=alpha, warm_up=warm_up)

    methods = {'Standard FL': std_fl, 'Basic DP-FL': dp_fl, 'EC-DP-FL': ecdp_fl}
    histories = {}

    for name, method in methods.items():
        print(f"\n--- Training {name} ---")
        acc_list = []
        for r in range(num_rounds):
            method.train_round(client_loaders, epochs=local_epochs)
            acc = method.test_accuracy(test_loader)
            acc_list.append(acc)
            print(f"Round {r+1}: {acc:.2f}%")
        histories[name] = acc_list

    # Evaluate comprehensive metrics
    metrics = compare_methods_comprehensive(std_fl, dp_fl, ecdp_fl, test_loader, device,
                                            num_classes=num_classes, class_names=class_names)

    if plot:
        plt.figure(figsize=(10,6))
        for name, acc in histories.items():
            plt.plot(range(1, len(acc)+1), acc, marker='o', label=name)
        plt.xlabel('Federation Round')
        plt.ylabel('Accuracy (%)')
        title = f'Convergence ({mode}={eps}) - {dataset}'
        if dirichlet_alpha is not None:
            title += f' (Dirichlet α={dirichlet_alpha})'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/convergence_{mode}_{eps}_{dataset}_seed{seed}.png', dpi=150)
        plt.show()

    return metrics, histories, test_loader

def tune_correction_params(per_round_epsilon=None, target_epsilon=None, clip_norm=2.3,
                           num_rounds=20, device='cpu',
                           c_values=[1.5, 2.0, 2.5],
                           alpha_values=[0.6, 0.7, 0.8],
                           warm_up_values=[0],
                           participation_rate=0.5,
                           seed=42, dataset='skin', dirichlet_alpha=None,
                           batch_size=32, local_epochs=3):
    best_acc = -1
    best_params = {}
    for c in c_values:
        for alpha in alpha_values:
            for warm_up in warm_up_values:
                print(f"\n--- Testing c={c}, α={alpha}, warm_up={warm_up} ---")
                _, histories, _ = run_comparison(
                    per_round_epsilon, target_epsilon, clip_norm, num_rounds, device,
                    c, alpha, warm_up, participation_rate, seed=seed, plot=False,
                    dataset=dataset, dirichlet_alpha=dirichlet_alpha,
                    batch_size=batch_size, local_epochs=local_epochs)
                acc = histories['EC-DP-FL'][-1]
                print(f"Final EC-DP-FL accuracy: {acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'c': c, 'alpha': alpha, 'warm_up': warm_up}
    print(f"\nBest params: {best_params} with accuracy {best_acc:.2f}%")
    return best_params

def run_tradeoff(epsilon_values, clip_norm, num_rounds=20, num_trials=3,
                 device='cpu', base_seed=42, mode='per_round',
                 dataset='skin', dirichlet_alpha=None, batch_size=32, local_epochs=3):
    print("\n" + "="*70)
    print(f"PRIVACY‑UTILITY TRADEOFF ANALYSIS ({mode} ε) - {dataset}")
    print("="*70)

    basic_means, ecdp_means = [], []
    basic_stds, ecdp_stds = [], []

    for eps in epsilon_values:
        print(f"\n--- {mode} ε = {eps} ---")
        basic_accs, ecdp_accs = [], []
        for trial in range(num_trials):
            seed = base_seed + trial
            print(f"  Trial {trial+1}/{num_trials} (seed={seed})")
            set_seed(seed)
            # Load data with Dirichlet and batch size
            client_loaders, test_loader, model_class, _, _ = get_dataset_components(
                dataset, num_clients=3, batch_size=batch_size, alpha=dirichlet_alpha, seed=seed)

            # Heuristic correction parameters based on ε
            if eps <= 0.5:
                c, alpha, warm_up = 1.5, 0.6, 0
            else:
                c, alpha, warm_up = 2.5, 0.8, 0

            if mode == 'per_round':
                dp = BasicDPFL(3, model_class, device, participation_rate=0.5,
                               epsilon=eps, clip_norm=clip_norm)
                ec = ECDPFL(3, model_class, device, participation_rate=0.5,
                            epsilon=eps, clip_norm=clip_norm,
                            c=c, alpha=alpha, warm_up=warm_up)
            else:
                dp = BasicDPFL(3, model_class, device, participation_rate=0.5,
                               epsilon=None, target_epsilon=eps, max_rounds=num_rounds,
                               clip_norm=clip_norm)
                ec = ECDPFL(3, model_class, device, participation_rate=0.5,
                            epsilon=None, target_epsilon=eps, max_rounds=num_rounds,
                            clip_norm=clip_norm,
                            c=c, alpha=alpha, warm_up=warm_up)

            for r in range(num_rounds):
                dp.train_round(client_loaders, epochs=local_epochs)
                ec.train_round(client_loaders, epochs=local_epochs)

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
    # Standard FL accuracy (run once with same settings)
    set_seed(base_seed)
    client_loaders, test_loader, model_class, _, _ = get_dataset_components(
        dataset, num_clients=3, batch_size=batch_size, alpha=dirichlet_alpha, seed=base_seed)
    std_fl = StandardFL(3, model_class, device, participation_rate=0.5)
    for r in range(num_rounds):
        std_fl.train_round(client_loaders, epochs=local_epochs)
    std_acc = std_fl.test_accuracy(test_loader)
    plt.axhline(y=std_acc, color='g', linestyle='--', label='Standard FL')
    plt.xscale('log')
    plt.xlabel(f'Privacy budget ε ({mode})')
    plt.ylabel('Accuracy (%)')
    title = f'Privacy‑Utility Tradeoff ({mode} ε) - {dataset}'
    if dirichlet_alpha is not None:
        title += f' (Dirichlet α={dirichlet_alpha})'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/tradeoff_{mode}_{dataset}_{dirichlet_alpha}.png', dpi=150)
    plt.show()

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
    parser.add_argument('--clip_norm', type=float, default=2.3,
                        help='Clipping norm (suggested from analyze_gradients.py)')
    parser.add_argument('--rounds', type=int, default=20,
                        help='Number of federation rounds')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # New parameters
    parser.add_argument('--participation_rate', type=float, default=0.5,
                        help='Fraction of clients participating each round')
    parser.add_argument('--dirichlet_alpha', type=float, default=None,
                        help='Dirichlet concentration parameter for non‑IID split (e.g., 0.1, 0.5). None = IID.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for local training and testing')
    parser.add_argument('--local_epochs', type=int, default=3,
                        help='Number of local training epochs per round')
    # Correction parameters (used if mode=comparison)
    parser.add_argument('--c', type=float, default=2.5)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--warm_up', type=int, default=0, help='Warm‑up rounds before applying correction (0 = none)')
    args = parser.parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Ensure exactly one of per_round_epsilon or target_epsilon is set for comparison/tune
    if args.mode in ['comparison', 'tune']:
        assert (args.per_round_epsilon is not None) ^ (args.target_epsilon is not None), \
            "Exactly one of --per_round_epsilon or --target_epsilon must be provided."

    if args.mode == 'comparison':
        run_comparison(
            per_round_epsilon=args.per_round_epsilon,
            target_epsilon=args.target_epsilon,
            clip_norm=args.clip_norm,
            num_rounds=args.rounds,
            device=device,
            c=args.c,
            alpha=args.alpha,
            warm_up=args.warm_up,
            participation_rate=args.participation_rate,
            seed=args.seed,
            dataset=args.dataset,
            dirichlet_alpha=args.dirichlet_alpha,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs
        )
    elif args.mode == 'tradeoff':
        epsilon_list = [0.1, 0.2, 0.5, 1.0, 2.0]
        run_tradeoff(
            epsilon_list,
            args.clip_norm,
            args.rounds,
            num_trials=3,
            device=device,
            base_seed=args.seed,
            mode='per_round',
            dataset=args.dataset,
            dirichlet_alpha=args.dirichlet_alpha,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs
        )
    elif args.mode == 'tune':
        tune_correction_params(
            per_round_epsilon=args.per_round_epsilon,
            target_epsilon=args.target_epsilon,
            clip_norm=args.clip_norm,
            num_rounds=args.rounds,
            device=device,
            participation_rate=args.participation_rate,
            seed=args.seed,
            dataset=args.dataset,
            dirichlet_alpha=args.dirichlet_alpha,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs
        )