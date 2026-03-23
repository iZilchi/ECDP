import argparse
import torch
import sys
import os
import random
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import get_skin_cancer_dataloaders
from utils.chest_xray_loader import get_chest_xray_dataloaders
from models.medium_cnn import MediumCNN
from models.chest_xray_cnn import ChestXRayCNN

from core.federated_learning import FederatedLearningBase as StandardFL
from core.dpfl import BasicDPFL, ECDPFL
from utils.metrics import ComprehensiveMetrics
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

def compute_model_size(model):
    """Return model size in MB."""
    param_size = sum(p.numel() for p in model.parameters())
    buffer_size = sum(b.numel() for b in model.buffers())
    total_params = param_size + buffer_size
    return total_params * 4 / (1024 ** 2)  # assuming float32 = 4 bytes

def inference_latency(model, test_loader, device, num_batches=10):
    """Average inference time per batch (ms)."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_batches:
                break
            data = data.to(device)
            start = time.time()
            _ = model(data)
            times.append((time.time() - start) * 1000)  # ms
    return np.mean(times)

def convergence_round(accuracy_history, final_acc, threshold=0.95):
    """Round when accuracy reaches threshold * final_acc for first time."""
    target = final_acc * threshold
    for i, acc in enumerate(accuracy_history):
        if acc >= target:
            return i + 1  # 1-indexed
    return len(accuracy_history)

def run_single_experiment(method_class, num_clients, model_class, device,
                          client_loaders, test_loader, num_rounds,
                          method_kwargs, class_names):
    """
    Train a single method and return all metrics.
    method_kwargs: dict passed to method constructor.
    class_names: list of class names for the dataset.
    """
    method = method_class(num_clients, model_class, device, **method_kwargs)
    accuracy_history = []
    round_times = []

    start_total = time.time()
    for r in range(num_rounds):
        start_round = time.time()
        method.train_round(client_loaders, epochs=2)
        round_times.append(time.time() - start_round)
        acc = method.test_accuracy(test_loader)
        accuracy_history.append(acc)
        # Print per-round accuracy
        print(f"Round {r+1}: {acc:.2f}%")
    total_time = time.time() - start_total

    # Final metrics on test set
    metrics = ComprehensiveMetrics(num_classes=len(class_names), class_names=class_names)
    final_metrics = metrics.compute_all_metrics(method.global_model, test_loader, device)

    # System metrics
    model_size = compute_model_size(method.global_model)
    infer_lat = inference_latency(method.global_model, test_loader, device)
    conv_round = convergence_round(accuracy_history, accuracy_history[-1])

    return {
        'accuracy_history': accuracy_history,
        'final_accuracy': accuracy_history[-1],
        'final_precision': final_metrics['precision'],
        'final_recall': final_metrics['recall'],
        'final_f1': final_metrics['f1_score'],
        'final_auc': final_metrics['auc_roc'],
        'confusion_matrix': final_metrics['confusion_matrix'],
        'total_time': total_time,
        'round_times': round_times,
        'convergence_round': conv_round,
        'inference_latency': infer_lat,
        'model_size': model_size,
    }

def run_comparison(per_round_epsilon=None, target_epsilon=None, clip_norm=3.5, num_rounds=20,
                   device='cpu', c=None, alpha_corr=None, seed=42, plot=True, dataset='skin',
                   alpha_data=None, num_clients=3, participation_rate=0.5):
    """
    Run full comparison with a single seed (given by --seed).
    If c or alpha_corr are None, they are chosen heuristically based on epsilon.
    """
    set_seed(seed)

    if per_round_epsilon is not None:
        mode = "per‑round ε"
        eps = per_round_epsilon
    else:
        mode = "total ε"
        eps = target_epsilon

    # Auto-select correction parameters if not provided
    if c is None:
        c = 1.5 if eps <= 0.5 else 2.5
    if alpha_corr is None:
        alpha_corr = 0.6 if eps <= 0.5 else 0.8

    print(f"\n{'='*60}")
    print(f"COMPARISON: {mode}={eps} over {num_rounds} rounds, clip_norm={clip_norm}, c={c}, α={alpha_corr}, seed={seed}, dataset={dataset}, alpha_data={alpha_data}, num_clients={num_clients}")
    print('='*60)

    # Load data
    client_loaders, test_loader, model_class, num_classes, class_names = get_dataset_components(
        dataset, num_clients=num_clients, alpha=alpha_data, seed=seed
    )

    # Prepare method kwargs
    base_kwargs = {
        'clip_norm': clip_norm,
        'participation_rate': participation_rate
    }
    if per_round_epsilon is not None:
        base_kwargs['epsilon'] = per_round_epsilon
    else:
        base_kwargs['target_epsilon'] = target_epsilon
        base_kwargs['max_rounds'] = num_rounds

    # Standard FL
    std_kwargs = base_kwargs.copy()
    std_kwargs.pop('epsilon', None)
    std_kwargs.pop('target_epsilon', None)
    std_kwargs.pop('max_rounds', None)
    std_kwargs.pop('clip_norm', None)
    print("\n--- Training Standard FL ---")
    std_res = run_single_experiment(StandardFL, num_clients, model_class, device,
                                    client_loaders, test_loader, num_rounds, std_kwargs,
                                    class_names)

    # Basic DP-FL
    dp_kwargs = base_kwargs.copy()
    print("\n--- Training Basic DP-FL ---")
    dp_res = run_single_experiment(BasicDPFL, num_clients, model_class, device,
                                   client_loaders, test_loader, num_rounds, dp_kwargs,
                                   class_names)

    # EC-DP-FL
    ecdp_kwargs = base_kwargs.copy()
    ecdp_kwargs['c'] = c
    ecdp_kwargs['alpha'] = alpha_corr
    print("\n--- Training EC-DP-FL ---")
    ecdp_res = run_single_experiment(ECDPFL, num_clients, model_class, device,
                                     client_loaders, test_loader, num_rounds, ecdp_kwargs,
                                     class_names)

    # Print results table
    print("\n" + "="*70)
    print("FINAL RESULTS (seed = {})".format(seed))
    print("="*70)
    for name, res in [("Standard FL", std_res), ("Basic DP-FL", dp_res), ("EC-DP-FL", ecdp_res)]:
        print(f"\n{name}:")
        print(f"  Accuracy:       {res['final_accuracy']:.2f}%")
        print(f"  Precision:      {res['final_precision']:.2f}%")
        print(f"  Recall:         {res['final_recall']:.2f}%")
        print(f"  F1-Score:       {res['final_f1']:.2f}%")
        print(f"  AUC-ROC:        {res['final_auc']:.2f}%")
        print(f"  Total Time (s): {res['total_time']:.2f}")
        print(f"  Conv Round:     {res['convergence_round']}")
        print(f"  Inf Latency (ms): {res['inference_latency']:.2f}")
        print(f"  Model Size (MB): {res['model_size']:.2f}")

    # Utility analysis
    improvement = ecdp_res['final_accuracy'] - dp_res['final_accuracy']
    dp_loss = std_res['final_accuracy'] - dp_res['final_accuracy']
    recovery = (improvement / dp_loss * 100) if dp_loss > 0 else 0
    print(f"\n🎯 UTILITY ANALYSIS (Eq. 9 & 10):")
    print(f"  Improvement: {improvement:.2f}%")
    print(f"  Recovery Rate: {recovery:.1f}%")

    # Save confusion matrices
    metrics = ComprehensiveMetrics(num_classes=num_classes, class_names=class_names)
    os.makedirs('results', exist_ok=True)
    metrics.plot_confusion_matrix(std_res['confusion_matrix'], "Standard FL",
                                  f"results/confusion_matrix_std_{dataset}_{eps}_clients{num_clients}.png")
    metrics.plot_confusion_matrix(dp_res['confusion_matrix'], "Basic DP-FL",
                                  f"results/confusion_matrix_dp_{dataset}_{eps}_clients{num_clients}.png")
    metrics.plot_confusion_matrix(ecdp_res['confusion_matrix'], "EC-DP-FL",
                                  f"results/confusion_matrix_ecdp_{dataset}_{eps}_clients{num_clients}.png")

    # Plot convergence
    if plot:
        plt.figure(figsize=(10,6))
        rounds = range(1, num_rounds+1)
        plt.plot(rounds, std_res['accuracy_history'], 'b-', label='Standard FL')
        plt.plot(rounds, dp_res['accuracy_history'], 'r-', label='Basic DP-FL')
        plt.plot(rounds, ecdp_res['accuracy_history'], 'g-', label='EC-DP-FL')
        plt.xlabel('Federation Round')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Convergence ({mode}={eps}) - {dataset} (α_data={alpha_data})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/convergence_{mode}_{eps}_{dataset}_alpha{alpha_data}_clients{num_clients}.png', dpi=150)
        plt.show()

    return {'std': std_res, 'dp': dp_res, 'ecdp': ecdp_res}

def run_ablation(per_round_epsilon=None, target_epsilon=None, clip_norm=3.5, num_rounds=20,
                 device='cpu', seed=42, dataset='skin', alpha_data=None, num_clients=3,
                 participation_rate=0.5):
    """
    Run ablation study: DP only, DP+Extreme Clipping, DP+Smoothing, Full EC-DP-FL.
    """
    set_seed(seed)

    if per_round_epsilon is not None:
        mode = "per‑round ε"
        eps = per_round_epsilon
    else:
        mode = "total ε"
        eps = target_epsilon

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: {mode}={eps} over {num_rounds} rounds, clip_norm={clip_norm}, dataset={dataset}, alpha_data={alpha_data}, num_clients={num_clients}, seed={seed}")
    print('='*70)

    # Load data
    client_loaders, test_loader, model_class, num_classes, class_names = get_dataset_components(
        dataset, num_clients=num_clients, alpha=alpha_data, seed=seed
    )

    # Base kwargs
    base_kwargs = {
        'clip_norm': clip_norm,
        'participation_rate': participation_rate
    }
    if per_round_epsilon is not None:
        base_kwargs['epsilon'] = per_round_epsilon
    else:
        base_kwargs['target_epsilon'] = target_epsilon
        base_kwargs['max_rounds'] = num_rounds

    # Monkey patch ErrorCorrection to support flags (if not already done)
    from core.error_correction import ErrorCorrection
    original_apply = ErrorCorrection.apply
    def new_apply(self, noisy_update, alpha, c, use_clipping=None, use_smoothing=None):
        # If instance flags exist, use them; otherwise use the passed ones (default True)
        if hasattr(self, 'use_clipping'):
            use_clipping = self.use_clipping
        else:
            use_clipping = True if use_clipping is None else use_clipping
        if hasattr(self, 'use_smoothing'):
            use_smoothing = self.use_smoothing
        else:
            use_smoothing = True if use_smoothing is None else use_smoothing
        return original_apply(self, noisy_update, alpha, c, use_clipping, use_smoothing)
    ErrorCorrection.apply = new_apply

    # Define partial correction classes
    class DPWithClippingOnly(ECDPFL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.error_correction.use_smoothing = False

    class DPWithSmoothingOnly(ECDPFL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.error_correction.use_clipping = False

    # Run each configuration
    configs = {
        'DP only': BasicDPFL,
        'DP + Extreme Clipping': DPWithClippingOnly,
        'DP + Smoothing': DPWithSmoothingOnly,
        'Full EC-DP-FL': ECDPFL
    }

    results = {}
    for name, cls in configs.items():
        print(f"\n--- Running {name} ---")
        kwargs = base_kwargs.copy()
        if name != 'DP only':
            # For correction methods, we need c and alpha (use heuristics based on epsilon)
            if eps is not None:
                if eps <= 0.5:
                    c, alpha = 1.5, 0.6
                else:
                    c, alpha = 2.5, 0.8
            else:
                c, alpha = 2.5, 0.8  # default
            kwargs['c'] = c
            kwargs['alpha'] = alpha
        res = run_single_experiment(cls, num_clients, model_class, device,
                                    client_loaders, test_loader, num_rounds, kwargs,
                                    class_names)
        results[name] = res

    # Print results table
    print("\n" + "="*70)
    print("ABLATION RESULTS (seed = {})".format(seed))
    print("="*70)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {res['final_accuracy']:.2f}%")
        print(f"  Precision: {res['final_precision']:.2f}%")
        print(f"  Recall: {res['final_recall']:.2f}%")
        print(f"  F1: {res['final_f1']:.2f}%")
        print(f"  AUC: {res['final_auc']:.2f}%")
        print(f"  Conv Round: {res['convergence_round']}")
        print(f"  Total Time: {res['total_time']:.2f}s")
        print(f"  Inf Latency: {res['inference_latency']:.2f}ms")
        print(f"  Model Size: {res['model_size']:.2f}MB")

    # Plot convergence for all configurations
    plt.figure(figsize=(10,6))
    rounds = range(1, num_rounds+1)
    colors = {'DP only': 'r', 'DP + Extreme Clipping': 'orange', 'DP + Smoothing': 'purple', 'Full EC-DP-FL': 'g'}
    for name, res in results.items():
        plt.plot(rounds, res['accuracy_history'], label=name, color=colors.get(name, 'k'))
    plt.xlabel('Federation Round')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Ablation Study: {mode}={eps} - {dataset}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/ablation_{mode}_{eps}_{dataset}_alpha{alpha_data}_clients{num_clients}.png', dpi=150)
    plt.show()

    return results

def run_tradeoff(epsilon_values, clip_norm, num_rounds=20, device='cpu', base_seed=42,
                 mode='per_round', dataset='skin', alpha_data=None, num_clients=3,
                 participation_rate=0.5):
    """
    Run privacy‑utility tradeoff for multiple epsilon values, single seed.
    """
    print("\n" + "="*70)
    print(f"PRIVACY‑UTILITY TRADEOFF ANALYSIS ({mode} ε) - {dataset} (alpha_data={alpha_data}, num_clients={num_clients}, seed={base_seed})")
    print("="*70)

    results_basic = []
    results_ecdp = []
    for eps in epsilon_values:
        print(f"\n--- ε = {eps} ---")
        # Correction parameters chosen heuristically; they can be overridden by --c and --alpha_corr,
        # but we are not passing them, so they'll be auto-selected inside run_comparison.
        # However, run_comparison expects c and alpha_corr as arguments; we pass None to trigger auto.
        comp_res = run_comparison(per_round_epsilon=eps if mode=='per_round' else None,
                                  target_epsilon=eps if mode=='total' else None,
                                  clip_norm=clip_norm, num_rounds=num_rounds,
                                  device=device, c=None, alpha_corr=None,
                                  seed=base_seed, plot=False, dataset=dataset,
                                  alpha_data=alpha_data, num_clients=num_clients,
                                  participation_rate=participation_rate)
        basic_acc = comp_res['dp']['final_accuracy']
        ecdp_acc = comp_res['ecdp']['final_accuracy']
        results_basic.append(basic_acc)
        results_ecdp.append(ecdp_acc)

    # Plot tradeoff
    plt.figure(figsize=(10,6))
    plt.plot(epsilon_values, results_basic, 'o-', label='Basic DP-FL')
    plt.plot(epsilon_values, results_ecdp, 's-', label='EC-DP-FL')
    # Compute Standard FL accuracy once
    set_seed(base_seed)
    client_loaders, test_loader, model_class, _, _ = get_dataset_components(
        dataset, num_clients=num_clients, alpha=alpha_data, seed=base_seed
    )
    std_fl = StandardFL(num_clients, model_class, device, participation_rate=participation_rate)
    for _ in range(num_rounds):
        std_fl.train_round(client_loaders, epochs=2)
    std_acc = std_fl.test_accuracy(test_loader)
    plt.axhline(y=std_acc, color='g', linestyle='--', label='Standard FL')
    plt.xscale('log')
    plt.xlabel(f'Privacy budget ε ({mode})')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Privacy‑Utility Tradeoff ({mode} ε) - {dataset}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/tradeoff_{mode}_{dataset}_alpha{alpha_data}_clients{num_clients}.png', dpi=150)
    plt.show()

    return results_basic, results_ecdp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['comparison', 'tradeoff', 'ablation'], default='comparison')
    parser.add_argument('--dataset', choices=['skin', 'chest'], default='skin')
    parser.add_argument('--per_round_epsilon', type=float, default=None)
    parser.add_argument('--target_epsilon', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=3.5)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    # Correction parameters – now default to None, meaning auto-select
    parser.add_argument('--c', type=float, default=None, help='Correction bound parameter (auto if None)')
    parser.add_argument('--alpha_corr', type=float, default=None, help='Smoothing coefficient (auto if None)')
    parser.add_argument('--alpha_data', type=float, default=None, help='Dirichlet concentration for non-IID')
    parser.add_argument('--num_clients', type=int, default=3)
    parser.add_argument('--participation_rate', type=float, default=0.5)
    args = parser.parse_args()

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
                       alpha_data=args.alpha_data, num_clients=args.num_clients,
                       participation_rate=args.participation_rate)
    elif args.mode == 'ablation':
        assert (args.per_round_epsilon is not None) ^ (args.target_epsilon is not None), \
            "Exactly one of --per_round_epsilon or --target_epsilon must be provided."
        run_ablation(args.per_round_epsilon, args.target_epsilon, args.clip_norm, args.rounds, device,
                     seed=args.seed, dataset=args.dataset, alpha_data=args.alpha_data,
                     num_clients=args.num_clients, participation_rate=args.participation_rate)
    elif args.mode == 'tradeoff':
        epsilon_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        run_tradeoff(epsilon_list, args.clip_norm, args.rounds, device=device, base_seed=args.seed,
                     mode='per_round', dataset=args.dataset, alpha_data=args.alpha_data,
                     num_clients=args.num_clients, participation_rate=args.participation_rate)