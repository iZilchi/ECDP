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

# ---------- Helper functions ----------
def get_model_constructor(dataset):
    if dataset == 'skin_cancer':
        return MediumCNN
    elif dataset == 'chest_xray':
        return ChestXRayCNN
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_dataloaders(dataset, num_clients, batch_size, alpha, seed):
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

def get_class_info(dataset):
    if dataset == 'skin_cancer':
        return 7, ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    else:
        return 2, ['Normal', 'Pneumonia']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Main experiment ----------
def run_comparison(dataset, num_clients, participation_rate, alpha_dirichlet,
                   per_round_epsilon=None, target_epsilon=None,
                   clip_norm=3.5, num_rounds=20, device='cpu',
                   c=1.5, alpha_smooth=0.6, seed=42, plot=True):
    print(f"\n{'='*60}")
    if per_round_epsilon is not None:
        mode = "per‑round ε"
        eps = per_round_epsilon
    else:
        mode = "total ε"
        eps = target_epsilon
    print(f"Dataset: {dataset}")
    print(f"Data distribution: {'IID' if alpha_dirichlet is None else f'Non-IID (Dirichlet α={alpha_dirichlet})'}")
    print(f"Client participation: {participation_rate*100:.0f}%")
    print(f"Number of clients: {num_clients}")
    print(f"COMPARISON: {mode}={eps} over {num_rounds} rounds")
    print(f"clip_norm={clip_norm}, c={c}, α={alpha_smooth}, seed={seed}")
    print('='*60)

    set_seed(seed)

    client_loaders, test_loader = get_dataloaders(
        dataset, num_clients=num_clients, batch_size=32,
        alpha=alpha_dirichlet, seed=seed
    )
    num_classes, class_names = get_class_info(dataset)
    model_constructor = get_model_constructor(dataset)

    std_fl = StandardFL(num_clients, model_constructor, device,
                        participation_rate=participation_rate)

    if per_round_epsilon is not None:
        dp_fl = BasicDPFL(num_clients, model_constructor, device,
                          epsilon=per_round_epsilon,
                          clip_norm=clip_norm,
                          participation_rate=participation_rate)
        ecdp_fl = ECDPFL(num_clients, model_constructor, device,
                         epsilon=per_round_epsilon,
                         clip_norm=clip_norm,
                         c=c, alpha=alpha_smooth, warm_up=0,
                         participation_rate=participation_rate)
    else:
        dp_fl = BasicDPFL(num_clients, model_constructor, device,
                          epsilon=None, target_epsilon=target_epsilon,
                          max_rounds=num_rounds, clip_norm=clip_norm,
                          participation_rate=participation_rate)
        ecdp_fl = ECDPFL(num_clients, model_constructor, device,
                         epsilon=None, target_epsilon=target_epsilon,
                         max_rounds=num_rounds, clip_norm=clip_norm,
                         c=c, alpha=alpha_smooth, warm_up=0,
                         participation_rate=participation_rate)

    # Train and track accuracy per round
    histories = {}
    for name, method in [('Standard FL', std_fl), ('Basic DP-FL', dp_fl), ('EC-DP-FL', ecdp_fl)]:
        print(f"\n--- Training {name} ---")
        acc_list = []
        for r in range(num_rounds):
            method.train_round(client_loaders, epochs=2)
            acc = method.test_accuracy(test_loader)
            acc_list.append(acc)
            print(f"Round {r+1}: {acc:.2f}%")
        histories[name] = acc_list

    # Compute metrics
    metrics = compare_methods_comprehensive(
        std_fl, dp_fl, ecdp_fl, test_loader, device,
        num_classes=num_classes, class_names=class_names
    )

    # Plot convergence
    if plot:
        plt.figure(figsize=(10,6))
        for name, acc in histories.items():
            plt.plot(range(1, len(acc)+1), acc, marker='o', label=name)
        plt.xlabel('Federation Round')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Convergence ({mode}={eps}) on {dataset}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/convergence_{dataset}_{mode}_{eps}_seed{seed}.png', dpi=150)
        plt.show()

    return metrics, histories

def tune_correction_params(dataset, num_clients, participation_rate, alpha_dirichlet,
                           per_round_epsilon=None, target_epsilon=None,
                           clip_norm=3.5, num_rounds=20, device='cpu',
                           c_values=[1.5, 2.0, 2.5],
                           alpha_values=[0.6, 0.7, 0.8],
                           seed=42):
    """Grid search over correction parameters (single run per combination)."""
    best_acc = -1
    best_params = {}
    for c in c_values:
        for alpha_smooth in alpha_values:
            print(f"\n--- Testing c={c}, α={alpha_smooth} ---")
            metrics, _ = run_comparison(
                dataset, num_clients, participation_rate, alpha_dirichlet,
                per_round_epsilon, target_epsilon,
                clip_norm, num_rounds, device,
                c, alpha_smooth, seed=seed, plot=False
            )
            acc = metrics['ecdp_fl']['accuracy']
            print(f"EC-DP-FL final accuracy: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_params = {'c': c, 'alpha': alpha_smooth}
    print(f"\nBest params: {best_params} with accuracy {best_acc:.2f}%")
    return best_params

def tune_clip_norm(dataset, num_clients, participation_rate, alpha_dirichlet,
                   per_round_epsilon=None, target_epsilon=None,
                   num_rounds=20, device='cpu',
                   c=1.5, alpha_smooth=0.6,
                   clip_norm_values=[0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0],
                   seed=42):
    """Grid search over clipping norms (single run per value)."""
    best_acc = -1
    best_norm = None
    for norm in clip_norm_values:
        print(f"\n--- Testing clip_norm={norm} ---")
        metrics, _ = run_comparison(
            dataset, num_clients, participation_rate, alpha_dirichlet,
            per_round_epsilon, target_epsilon,
            norm, num_rounds, device,
            c, alpha_smooth, seed=seed, plot=False
        )
        acc = metrics['ecdp_fl']['accuracy']
        print(f"EC-DP-FL final accuracy: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_norm = norm
    print(f"\nBest clipping norm: {best_norm} with accuracy {best_acc:.2f}%")
    return best_norm

def ablation_study(dataset, num_clients, participation_rate, alpha_dirichlet,
                   per_round_epsilon=None, target_epsilon=None,
                   clip_norm=3.5, num_rounds=20, device='cpu',
                   c=1.5, alpha_smooth=0.6, seed=42):
    """Run ablation: DP only, DP+clip, DP+EVC, DP+AGS, full ECDP-FL."""
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)
    print(f"Dataset: {dataset}")
    print(f"Data: {'IID' if alpha_dirichlet is None else f'Non-IID α={alpha_dirichlet}'}")
    print(f"Participation: {participation_rate*100:.0f}%, Clients: {num_clients}")
    print(f"Privacy: {'per-round ε='+str(per_round_epsilon) if per_round_epsilon else 'total ε='+str(target_epsilon)}")
    print("="*70)

    set_seed(seed)

    client_loaders, test_loader = get_dataloaders(
        dataset, num_clients=num_clients, batch_size=32,
        alpha=alpha_dirichlet, seed=seed
    )
    num_classes, class_names = get_class_info(dataset)
    model_constructor = get_model_constructor(dataset)

    # Custom DP classes for ablation
    class DPOnly(BasicDPFL):
        def _train_client_get_update(self, global_weights, dataloader, epochs):
            return super()._train_client_get_update(global_weights, dataloader, epochs)
    class DPClip(BasicDPFL):
        pass
    class DPEVC(BasicDPFL):
        def _aggregate_updates(self, client_updates):
            avg_update = super()._aggregate_updates(client_updates)
            from core.error_correction import ErrorCorrection
            ec = ErrorCorrection()
            corrected = ec.apply(avg_update, alpha=1.0, c=c, warm_up_rounds=0,
                                 use_clipping=True, use_smoothing=False)
            return corrected
    class DPAGS(BasicDPFL):
        def _aggregate_updates(self, client_updates):
            avg_update = super()._aggregate_updates(client_updates)
            from core.error_correction import ErrorCorrection
            ec = ErrorCorrection()
            corrected = ec.apply(avg_update, alpha=alpha_smooth, c=1.0, warm_up_rounds=0,
                                 use_clipping=False, use_smoothing=True)
            return corrected

    configs = {
        "DP only": DPOnly,
        "DP + clipping": DPClip,
        "DP + EVC": DPEVC,
        "DP + AGS": DPAGS,
        "ECDP-FL": lambda *args, **kwargs: ECDPFL(*args, c=c, alpha=alpha_smooth, warm_up=0, **kwargs)
    }

    results = {}
    for name, method_class in configs.items():
        print(f"\n--- {name} ---")
        method = method_class(num_clients, model_constructor, device,
                              epsilon=per_round_epsilon, target_epsilon=target_epsilon,
                              max_rounds=num_rounds, clip_norm=clip_norm,
                              participation_rate=participation_rate)
        for _ in range(num_rounds):
            method.train_round(client_loaders, epochs=2)
        acc = method.test_accuracy(test_loader)
        results[name] = acc
        print(f"Final accuracy: {acc:.2f}%")

    # Bar plot
    plt.figure(figsize=(10,6))
    names = list(results.keys())
    values = list(results.values())
    plt.bar(names, values)
    plt.ylabel('Accuracy (%)')
    plt.title('Ablation Study')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/ablation.png', dpi=150)
    plt.show()
    return results

def run_tradeoff(dataset, num_clients, participation_rate, alpha_dirichlet,
                 epsilon_values, clip_norm, num_rounds=10,
                 device='cpu', base_seed=42, mode='per_round'):
    """Privacy‑utility tradeoff (single run per epsilon)."""
    print("\n" + "="*70)
    print(f"PRIVACY‑UTILITY TRADEOFF ANALYSIS ({mode} ε) on {dataset}")
    print(f"Data distribution: {'IID' if alpha_dirichlet is None else f'Non-IID (Dirichlet α={alpha_dirichlet})'}")
    print(f"Client participation: {participation_rate*100:.0f}%")
    print("="*70)

    model_constructor = get_model_constructor(dataset)

    basic_accs = []
    ecdp_accs = []
    std_acc = None

    for eps in epsilon_values:
        print(f"\n--- {mode} ε = {eps} ---")
        set_seed(base_seed)

        client_loaders, test_loader = get_dataloaders(
            dataset, num_clients=num_clients, batch_size=32,
            alpha=alpha_dirichlet, seed=base_seed
        )

        # Correction parameters based on ε (warm_up=0)
        if eps <= 0.5:
            c, alpha_smooth = 1.5, 0.6
        else:
            c, alpha_smooth = 2.5, 0.8

        if mode == 'per_round':
            dp = BasicDPFL(num_clients, model_constructor, device,
                           epsilon=eps, clip_norm=clip_norm,
                           participation_rate=participation_rate)
            ec = ECDPFL(num_clients, model_constructor, device,
                        epsilon=eps, clip_norm=clip_norm,
                        c=c, alpha=alpha_smooth, warm_up=0,
                        participation_rate=participation_rate)
        else:
            dp = BasicDPFL(num_clients, model_constructor, device,
                           epsilon=None, target_epsilon=eps,
                           max_rounds=num_rounds, clip_norm=clip_norm,
                           participation_rate=participation_rate)
            ec = ECDPFL(num_clients, model_constructor, device,
                        epsilon=None, target_epsilon=eps,
                        max_rounds=num_rounds, clip_norm=clip_norm,
                        c=c, alpha=alpha_smooth, warm_up=0,
                        participation_rate=participation_rate)

        for r in range(num_rounds):
            dp.train_round(client_loaders, epochs=2)
            ec.train_round(client_loaders, epochs=2)

        basic_accs.append(dp.test_accuracy(test_loader))
        ecdp_accs.append(ec.test_accuracy(test_loader))
        print(f"Basic DP: {basic_accs[-1]:.2f}%")
        print(f"EC-DP:    {ecdp_accs[-1]:.2f}%")

        # Standard FL only once (use the first epsilon's setup)
        if std_acc is None:
            std_fl = StandardFL(num_clients, model_constructor, device,
                                participation_rate=participation_rate)
            for r in range(num_rounds):
                std_fl.train_round(client_loaders, epochs=2)
            std_acc = std_fl.test_accuracy(test_loader)
            print(f"Standard FL: {std_acc:.2f}% (constant)")

    # Plot tradeoff
    plt.figure(figsize=(10,6))
    plt.plot(epsilon_values, basic_accs, marker='o', label='Basic DP-FL')
    plt.plot(epsilon_values, ecdp_accs, marker='s', label='EC-DP-FL')
    plt.axhline(y=std_acc, color='g', linestyle='--', label='Standard FL')
    plt.xscale('log')
    plt.xlabel(f'Privacy budget ε ({mode})')
    plt.ylabel('Accuracy (%)')
    dist = 'IID' if alpha_dirichlet is None else f'Non-IID α={alpha_dirichlet}'
    plt.title(f'Privacy‑Utility Tradeoff on {dataset} ({dist})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/tradeoff_{dataset}_{mode}_{dist.replace(" ", "_")}.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['comparison', 'tradeoff', 'tune', 'ablation', 'tune_clip_norm'],
                        default='comparison')
    parser.add_argument('--dataset', choices=['skin_cancer', 'chest_xray'], default='skin_cancer')
    parser.add_argument('--per_round_epsilon', type=float, default=None)
    parser.add_argument('--target_epsilon', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=3.5)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--c', type=float, default=1.5)
    parser.add_argument('--alpha_smooth', type=float, default=0.6)
    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--full_participation', action='store_true')
    parser.add_argument('--num_clients', type=int, default=3)

    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🖥️ Using device: {device}")

    alpha_dirichlet = None if args.iid else 0.5
    participation_rate = 1.0 if args.full_participation else 0.5
    num_clients = args.num_clients

    if args.mode in ['comparison', 'tune', 'ablation', 'tune_clip_norm']:
        if args.per_round_epsilon is None and args.target_epsilon is None:
            parser.error(f"Mode '{args.mode}' requires either --per_round_epsilon or --target_epsilon")

    if args.mode == 'comparison':
        run_comparison(
            args.dataset, num_clients, participation_rate, alpha_dirichlet,
            args.per_round_epsilon, args.target_epsilon,
            args.clip_norm, args.rounds, device,
            args.c, args.alpha_smooth, seed=args.seed
        )
    elif args.mode == 'tradeoff':
        epsilon_list = [0.1, 0.2, 0.5, 1.0, 2.0]
        run_tradeoff(
            args.dataset, num_clients, participation_rate, alpha_dirichlet,
            epsilon_list, args.clip_norm, args.rounds,
            device=device, base_seed=args.seed, mode='per_round'
        )
    elif args.mode == 'tune':
        tune_correction_params(
            args.dataset, num_clients, participation_rate, alpha_dirichlet,
            args.per_round_epsilon, args.target_epsilon,
            args.clip_norm, args.rounds, device,
            seed=args.seed
        )
    elif args.mode == 'ablation':
        ablation_study(
            args.dataset, num_clients, participation_rate, alpha_dirichlet,
            args.per_round_epsilon, args.target_epsilon,
            args.clip_norm, args.rounds, device,
            args.c, args.alpha_smooth, seed=args.seed
        )
    elif args.mode == 'tune_clip_norm':
        tune_clip_norm(
            args.dataset, num_clients, participation_rate, alpha_dirichlet,
            args.per_round_epsilon, args.target_epsilon,
            args.rounds, device,
            args.c, args.alpha_smooth,
            seed=args.seed
        )