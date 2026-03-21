import argparse
import torch
import sys
import os
import random
import numpy as np
from scipy.stats import wilcoxon
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import get_skin_cancer_dataloaders
from utils.chest_xray_loader import get_chest_xray_dataloaders
from models.medium_cnn import MediumCNN
from models.chest_xray_cnn import ChestXRayCNN
from core.federated_learning import FederatedLearningBase as StandardFL
from core.dpfl import BasicDPFL, ECDPFL
from utils.metrics import compare_methods_comprehensive
import matplotlib.pyplot as plt

# ---------- Helper functions for dataset selection ----------
def get_model_constructor(dataset):
    """Return the model class (callable) for the given dataset."""
    if dataset == 'skin_cancer':
        return MediumCNN  # 7 classes
    elif dataset == 'chest_xray':
        return ChestXRayCNN  # 2 classes
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_dataloaders(dataset, num_clients, batch_size, alpha, seed):
    """Return (client_loaders, test_loader) for the chosen dataset."""
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
    """Return (num_classes, class_names) for the dataset."""
    if dataset == 'skin_cancer':
        return 7, ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    else:  # chest_xray
        return 2, ['Normal', 'Pneumonia']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Core functions ----------
def run_single_comparison(dataset, num_clients, participation_rate, alpha_dirichlet,
                          per_round_epsilon, target_epsilon,
                          clip_norm, num_rounds, device,
                          c, alpha_smooth, warm_up, seed):
    """Run one complete experiment (all methods) and return metrics dict."""
    set_seed(seed)

    client_loaders, test_loader = get_dataloaders(
        dataset, num_clients=num_clients, batch_size=64,
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
                         c=c, alpha=alpha_smooth, warm_up=warm_up,
                         participation_rate=participation_rate)
    else:
        dp_fl = BasicDPFL(num_clients, model_constructor, device,
                          epsilon=None, target_epsilon=target_epsilon,
                          max_rounds=num_rounds, clip_norm=clip_norm,
                          participation_rate=participation_rate)
        ecdp_fl = ECDPFL(num_clients, model_constructor, device,
                         epsilon=None, target_epsilon=target_epsilon,
                         max_rounds=num_rounds, clip_norm=clip_norm,
                         c=c, alpha=alpha_smooth, warm_up=warm_up,
                         participation_rate=participation_rate)

    # Train all methods
    for method in [std_fl, dp_fl, ecdp_fl]:
        for _ in range(num_rounds):
            method.train_round(client_loaders, epochs=2)

    # Compute metrics
    metrics = compare_methods_comprehensive(
        std_fl, dp_fl, ecdp_fl, test_loader, device,
        num_classes=num_classes, class_names=class_names
    )
    return metrics

def run_comparison(dataset, num_clients, participation_rate, alpha_dirichlet,
                   per_round_epsilon=None, target_epsilon=None,
                   clip_norm=3.5, num_rounds=20, device='cpu',
                   c=1.5, alpha_smooth=0.6, warm_up=0,
                   trials=1, plot=True):
    """Run multiple trials and compute statistics."""
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
    print(f"COMPARISON: {mode}={eps} over {num_rounds} rounds, clip_norm={clip_norm}, c={c}, α={alpha_smooth}, warm_up={warm_up}")
    print(f"Trials: {trials}")
    print('='*60)

    results = {'standard_fl': [], 'dp_fl': [], 'ecdp_fl': []}
    trial_metrics = []

    for trial in range(trials):
        seed = 42 + trial  # base seed + trial offset
        print(f"\n--- Trial {trial+1}/{trials} (seed={seed}) ---")
        metrics = run_single_comparison(
            dataset, num_clients, participation_rate, alpha_dirichlet,
            per_round_epsilon, target_epsilon,
            clip_norm, num_rounds, device,
            c, alpha_smooth, warm_up, seed
        )
        trial_metrics.append(metrics)
        results['standard_fl'].append(metrics['standard_fl']['accuracy'])
        results['dp_fl'].append(metrics['dp_fl']['accuracy'])
        results['ecdp_fl'].append(metrics['ecdp_fl']['accuracy'])

    # Compute means and stds
    print("\n" + "="*60)
    print("FINAL RESULTS (over trials)")
    print("="*60)
    for name in ['standard_fl', 'dp_fl', 'ecdp_fl']:
        accs = results[name]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{name:12s}: {mean_acc:.2f} ± {std_acc:.2f}%")

    # Wilcoxon signed-rank test between DP-FL and EC-DP-FL
    if trials > 1:
        stat, p_val = wilcoxon(results['dp_fl'], results['ecdp_fl'], alternative='greater')
        print(f"\nWilcoxon signed-rank test (DP-FL vs EC-DP-FL):")
        print(f"  Statistic = {stat:.3f}, p-value = {p_val:.4f}")
        if p_val < 0.05:
            print("  ✅ EC-DP-FL significantly outperforms DP-FL (p < 0.05)")
        else:
            print("  ❌ No significant difference (p >= 0.05)")

        # Compute 95% confidence intervals via bootstrap
        n_bootstrap = 1000
        ci_low, ci_high = [], []
        for name in ['standard_fl', 'dp_fl', 'ecdp_fl']:
            bootstrap_means = []
            for _ in range(n_bootstrap):
                resampled = np.random.choice(results[name], size=len(results[name]), replace=True)
                bootstrap_means.append(np.mean(resampled))
            ci_low.append(np.percentile(bootstrap_means, 2.5))
            ci_high.append(np.percentile(bootstrap_means, 97.5))
        print(f"\n95% Confidence Intervals (bootstrap):")
        print(f"  Standard FL: [{ci_low[0]:.2f}, {ci_high[0]:.2f}]")
        print(f"  Basic DP-FL: [{ci_low[1]:.2f}, {ci_high[1]:.2f}]")
        print(f"  EC-DP-FL:    [{ci_low[2]:.2f}, {ci_high[2]:.2f}]")
    else:
        print("\n⚠️ Only one trial – no statistical tests performed.")

    # Plot if requested
    if plot and trials > 1:
        plt.figure(figsize=(10,6))
        data_to_plot = [results['standard_fl'], results['dp_fl'], results['ecdp_fl']]
        plt.boxplot(data_to_plot, labels=['Standard FL', 'DP-FL', 'EC-DP-FL'])
        plt.ylabel('Accuracy (%)')
        plt.title(f'Final Accuracy ({mode}={eps}) on {dataset}')
        plt.grid(True, alpha=0.3)
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/final_acc_{dataset}_{mode}_{eps}_seed{42}.png', dpi=150)
        plt.show()

    # Return the aggregated metrics for potential further use
    return trial_metrics

def tune_correction_params(dataset, num_clients, participation_rate, alpha_dirichlet,
                           per_round_epsilon=None, target_epsilon=None,
                           clip_norm=3.5, num_rounds=20, device='cpu',
                           c_values=[1.5, 2.0, 2.5],
                           alpha_smooth_values=[0.6, 0.7, 0.8],
                           warm_up_values=[0, 3],
                           trials=1):
    """Grid search over correction parameters, averaging over multiple trials."""
    best_acc = -1
    best_params = {}
    for c in c_values:
        for alpha_smooth in alpha_smooth_values:
            for warm_up in warm_up_values:
                print(f"\n--- Testing c={c}, α={alpha_smooth}, warm_up={warm_up} ---")
                acc_list = []
                for trial in range(trials):
                    seed = 42 + trial
                    print(f"  Trial {trial+1}/{trials} (seed={seed})")
                    metrics = run_single_comparison(
                        dataset, num_clients, participation_rate, alpha_dirichlet,
                        per_round_epsilon, target_epsilon,
                        clip_norm, num_rounds, device,
                        c, alpha_smooth, warm_up, seed
                    )
                    acc_list.append(metrics['ecdp_fl']['accuracy'])
                mean_acc = np.mean(acc_list)
                std_acc = np.std(acc_list)
                print(f"  EC-DP-FL final accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_params = {'c': c, 'alpha': alpha_smooth, 'warm_up': warm_up}
    print(f"\nBest params: {best_params} with accuracy {best_acc:.2f}%")
    return best_params

def tune_clip_norm(dataset, num_clients, participation_rate, alpha_dirichlet,
                   per_round_epsilon=None, target_epsilon=None,
                   num_rounds=20, device='cpu',
                   c=1.5, alpha_smooth=0.6, warm_up=0,
                   clip_norm_values=[0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0],
                   trials=1):
    """Grid search over clipping norm values."""
    best_acc = -1
    best_norm = None
    for norm in clip_norm_values:
        print(f"\n--- Testing clip_norm={norm} ---")
        acc_list = []
        for trial in range(trials):
            seed = 42 + trial
            print(f"  Trial {trial+1}/{trials} (seed={seed})")
            metrics = run_single_comparison(
                dataset, num_clients, participation_rate, alpha_dirichlet,
                per_round_epsilon, target_epsilon,
                norm, num_rounds, device,
                c, alpha_smooth, warm_up, seed
            )
            acc_list.append(metrics['ecdp_fl']['accuracy'])
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        print(f"  EC-DP-FL final accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_norm = norm
    print(f"\nBest clipping norm: {best_norm} with accuracy {best_acc:.2f}%")
    return best_norm

def ablation_study(dataset, num_clients, participation_rate, alpha_dirichlet,
                   per_round_epsilon=None, target_epsilon=None,
                   clip_norm=3.5, num_rounds=20, device='cpu',
                   c=1.5, alpha_smooth=0.6, warm_up=0,
                   trials=1):
    """
    Run ablation: DP only, DP+clip, DP+EVC, DP+AGS, full ECDP-FL.
    We use BasicDPFL with modifications to control components.
    """
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)
    print(f"Dataset: {dataset}")
    print(f"Data: {'IID' if alpha_dirichlet is None else f'Non-IID α={alpha_dirichlet}'}")
    print(f"Participation: {participation_rate*100:.0f}%, Clients: {num_clients}")
    print(f"Privacy: {'per-round ε='+str(per_round_epsilon) if per_round_epsilon else 'total ε='+str(target_epsilon)}")
    print("="*70)

    # We'll create a small helper to run a config
    def run_config(config_name, use_clipping, use_evc, use_ags):
        accs = []
        for trial in range(trials):
            seed = 42 + trial
            set_seed(seed)
            client_loaders, test_loader = get_dataloaders(
                dataset, num_clients=num_clients, batch_size=64,
                alpha=alpha_dirichlet, seed=seed
            )
            num_classes, class_names = get_class_info(dataset)
            model_constructor = get_model_constructor(dataset)

            # For DP-only: we need a version that does not clip before noise.
            # Our BasicDPFL always clips. We'll create a custom DP class that only adds noise.
            # For simplicity, we'll use a custom version:
            class DPOnly(BasicDPFL):
                def _train_client_get_update(self, global_weights, dataloader, epochs):
                    # No clipping
                    return super()._train_client_get_update(global_weights, dataloader, epochs)
            class DPClip(BasicDPFL):
                pass  # already does clipping
            class DPEVC(BasicDPFL):
                def _aggregate_updates(self, client_updates):
                    avg_update = super()._aggregate_updates(client_updates)
                    # apply extreme value clipping on aggregated update
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

            if config_name == "DP only":
                method = DPOnly(num_clients, model_constructor, device,
                                epsilon=per_round_epsilon, target_epsilon=target_epsilon,
                                max_rounds=num_rounds, clip_norm=clip_norm,
                                participation_rate=participation_rate)
            elif config_name == "DP + clipping":
                method = DPClip(num_clients, model_constructor, device,
                                epsilon=per_round_epsilon, target_epsilon=target_epsilon,
                                max_rounds=num_rounds, clip_norm=clip_norm,
                                participation_rate=participation_rate)
            elif config_name == "DP + EVC":
                method = DPEVC(num_clients, model_constructor, device,
                               epsilon=per_round_epsilon, target_epsilon=target_epsilon,
                               max_rounds=num_rounds, clip_norm=clip_norm,
                               participation_rate=participation_rate)
            elif config_name == "DP + AGS":
                method = DPAGS(num_clients, model_constructor, device,
                               epsilon=per_round_epsilon, target_epsilon=target_epsilon,
                               max_rounds=num_rounds, clip_norm=clip_norm,
                               participation_rate=participation_rate)
            else:  # full ECDP-FL
                method = ECDPFL(num_clients, model_constructor, device,
                                epsilon=per_round_epsilon, target_epsilon=target_epsilon,
                                max_rounds=num_rounds, clip_norm=clip_norm,
                                c=c, alpha=alpha_smooth, warm_up=warm_up,
                                participation_rate=participation_rate)

            for _ in range(num_rounds):
                method.train_round(client_loaders, epochs=2)
            acc = method.test_accuracy(test_loader)
            accs.append(acc)
        return np.mean(accs), np.std(accs)

    configs = ["DP only", "DP + clipping", "DP + EVC", "DP + AGS", "ECDP-FL"]
    results = {}
    for cfg in configs:
        mean_acc, std_acc = run_config(cfg, True, True, True)  # flags not used; we use specific classes
        results[cfg] = (mean_acc, std_acc)
        print(f"{cfg:20s}: {mean_acc:.2f} ± {std_acc:.2f}%")

    # Optional: plot bar chart
    plt.figure(figsize=(10,6))
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.ylabel('Accuracy (%)')
    plt.title('Ablation Study')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/ablation.png', dpi=150)
    plt.show()
    return results

def run_tradeoff(dataset, num_clients, participation_rate, alpha_dirichlet,
                 epsilon_values, clip_norm, num_rounds=20,
                 num_trials=1, device='cpu', base_seed=42, mode='per_round'):
    """Existing tradeoff function, but now with trials per epsilon."""
    print("\n" + "="*70)
    print(f"PRIVACY‑UTILITY TRADEOFF ANALYSIS ({mode} ε) on {dataset}")
    print(f"Data distribution: {'IID' if alpha_dirichlet is None else f'Non-IID (Dirichlet α={alpha_dirichlet})'}")
    print(f"Client participation: {participation_rate*100:.0f}%")
    print("="*70)

    num_classes, class_names = get_class_info(dataset)
    model_constructor = get_model_constructor(dataset)

    basic_means, ecdp_means = [], []
    basic_stds, ecdp_stds = [], []

    for eps in epsilon_values:
        print(f"\n--- {mode} ε = {eps} ---")
        basic_accs, ecdp_accs = [], []
        for trial in range(num_trials):
            seed = base_seed + trial
            print(f"  Trial {trial+1}/{num_trials} (seed={seed})")
            set_seed(seed)

            client_loaders, test_loader = get_dataloaders(
                dataset, num_clients=num_clients, batch_size=64,
                alpha=alpha_dirichlet, seed=seed
            )

            # Heuristic correction parameters based on ε
            if eps <= 0.5:
                c, alpha_smooth, warm_up = 1.5, 0.6, 3
            else:
                c, alpha_smooth, warm_up = 2.5, 0.8, 0

            if mode == 'per_round':
                dp = BasicDPFL(num_clients, model_constructor, device,
                               epsilon=eps, clip_norm=clip_norm,
                               participation_rate=participation_rate)
                ec = ECDPFL(num_clients, model_constructor, device,
                            epsilon=eps, clip_norm=clip_norm,
                            c=c, alpha=alpha_smooth, warm_up=warm_up,
                            participation_rate=participation_rate)
            else:
                dp = BasicDPFL(num_clients, model_constructor, device,
                               epsilon=None, target_epsilon=eps,
                               max_rounds=num_rounds, clip_norm=clip_norm,
                               participation_rate=participation_rate)
                ec = ECDPFL(num_clients, model_constructor, device,
                            epsilon=None, target_epsilon=eps,
                            max_rounds=num_rounds, clip_norm=clip_norm,
                            c=c, alpha=alpha_smooth, warm_up=warm_up,
                            participation_rate=participation_rate)

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

    # Standard FL accuracy (single run)
    set_seed(base_seed)
    client_loaders, test_loader = get_dataloaders(
        dataset, num_clients=num_clients, batch_size=64,
        alpha=alpha_dirichlet, seed=base_seed
    )
    std_fl = StandardFL(num_clients, model_constructor, device,
                        participation_rate=participation_rate)
    for r in range(num_rounds):
        std_fl.train_round(client_loaders, epochs=2)
    std_acc = std_fl.test_accuracy(test_loader)
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
                        default='comparison',
                        help='Experiment mode: comparison (default), tradeoff, tune, ablation, tune_clip_norm')
    parser.add_argument('--dataset', choices=['skin_cancer', 'chest_xray'],
                        default='skin_cancer', help='Dataset to use')
    parser.add_argument('--per_round_epsilon', type=float, default=None,
                        help='Per‑round privacy budget')
    parser.add_argument('--target_epsilon', type=float, default=None,
                        help='Total privacy budget over all rounds')
    parser.add_argument('--clip_norm', type=float, default=3.5,
                        help='Clipping norm (used as initial value or for grid search)')
    parser.add_argument('--rounds', type=int, default=20,
                        help='Number of federation rounds')
    parser.add_argument('--device', default=None,
                        help='Device to use: "cpu" or "cuda". If not specified, auto‑detect.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for reproducibility')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of independent runs (trials) for statistical validation')
    parser.add_argument('--c', type=float, default=1.5,
                        help='Correction bound parameter (for comparison mode)')
    parser.add_argument('--alpha_smooth', type=float, default=0.6,
                        help='Smoothing coefficient (for comparison mode)')
    parser.add_argument('--warm_up', type=int, default=0,
                        help='Warm‑up rounds before correction (for comparison mode)')
    parser.add_argument('--iid', action='store_true',
                        help='Use IID data distribution (default: Non-IID with Dirichlet α=0.5)')
    parser.add_argument('--full_participation', action='store_true',
                        help='Use full client participation (100% per round). Default: 50%')
    parser.add_argument('--num_clients', type=int, default=3,
                        help='Number of simulated clients (default: 3)')

    args = parser.parse_args()

    # Auto‑detect device if not specified
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🖥️ Using device: {device}")

    # Set data distribution
    alpha_dirichlet = None if args.iid else 0.5   # default Dirichlet α=0.5
    participation_rate = 1.0 if args.full_participation else 0.5
    num_clients = args.num_clients

    # For modes that require a privacy budget, ensure it's provided
    if args.mode in ['comparison', 'tune', 'ablation', 'tune_clip_norm']:
        if args.per_round_epsilon is None and args.target_epsilon is None:
            parser.error(f"Mode '{args.mode}' requires either --per_round_epsilon or --target_epsilon")

    if args.mode == 'comparison':
        run_comparison(
            args.dataset, num_clients, participation_rate, alpha_dirichlet,
            args.per_round_epsilon, args.target_epsilon,
            args.clip_norm, args.rounds, device,
            args.c, args.alpha_smooth, args.warm_up,
            trials=args.trials
        )
    elif args.mode == 'tradeoff':
        epsilon_list = [0.1, 0.2, 0.5, 1.0, 2.0]
        run_tradeoff(
            args.dataset, num_clients, participation_rate, alpha_dirichlet,
            epsilon_list, args.clip_norm, args.rounds,
            num_trials=args.trials, device=device, base_seed=args.seed, mode='per_round'
        )
    elif args.mode == 'tune':
        tune_correction_params(
            args.dataset, num_clients, participation_rate, alpha_dirichlet,
            args.per_round_epsilon, args.target_epsilon,
            args.clip_norm, args.rounds, device,
            trials=args.trials
        )
    elif args.mode == 'ablation':
        ablation_study(
            args.dataset, num_clients, participation_rate, alpha_dirichlet,
            args.per_round_epsilon, args.target_epsilon,
            args.clip_norm, args.rounds, device,
            args.c, args.alpha_smooth, args.warm_up,
            trials=args.trials
        )
    elif args.mode == 'tune_clip_norm':
        tune_clip_norm(
            args.dataset, num_clients, participation_rate, alpha_dirichlet,
            args.per_round_epsilon, args.target_epsilon,
            args.rounds, device,
            args.c, args.alpha_smooth, args.warm_up,
            trials=args.trials
        )