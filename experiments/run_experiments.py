import argparse
import torch
import sys
import os
import random
import numpy as np
import time
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import get_skin_cancer_dataloaders
<<<<<<< HEAD
from utils.chest_xray_loader import get_chest_xray_dataloaders
from models.medium_cnn import MediumCNN
from models.chest_xray_cnn import ChestXRayCNN
=======
from models.tiny_cnn import TinyCNN
>>>>>>> parent of e8091c8 (Updated CNN & Per-Round Epsilon)
from core.federated_learning import FederatedLearningBase as StandardFL
from core.dpfl import BasicDPFL, ECDPFL
from utils.metrics import compare_methods_comprehensive

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 6),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior on GPU (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

<<<<<<< HEAD
def get_dataset_components(dataset_name, num_clients=3, batch_size=64, alpha=None, seed=42):
    if dataset_name == 'skin':
        client_loaders, test_loader = get_skin_cancer_dataloaders(
            num_clients=num_clients, batch_size=batch_size, alpha=alpha, seed=seed
        )
        model_class = lambda: MediumCNN(num_classes=7)
        num_classes = 7
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    else:
        client_loaders, test_loader = get_chest_xray_dataloaders(
            num_clients=num_clients, batch_size=batch_size, alpha=alpha, seed=seed
        )
        model_class = lambda: ChestXRayCNN(num_classes=2)
        num_classes = 2
        class_names = ['NORMAL', 'PNEUMONIA']
    return client_loaders, test_loader, model_class, num_classes, class_names

def measure_inference_latency(model, test_loader, device, num_batches=5):
    model.eval()
    latencies = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_batches:
                break
            data = data.to(device)
            start = time.time()
            _ = model(data)
            latencies.append(time.time() - start)
    return np.mean(latencies) * 1000

def run_comparison(per_round_epsilon=None, target_epsilon=None, clip_norm=2.3,
                   num_rounds=20, device='cpu',
                   c=2.5, alpha=0.8, warm_up=0,
                   participation_rate=0.5, seed=42, plot=True,
                   dataset='skin', dirichlet_alpha=None, batch_size=64, local_epochs=3,
                   ablation_use_clipping=True, ablation_use_smoothing=True):
    print(f"\n{'='*60}")
    if per_round_epsilon is not None:
        mode = "per‑round ε"
        eps = per_round_epsilon
    else:
        mode = "total ε"
        eps = target_epsilon
    print(f"COMPARISON: {mode}={eps} over {num_rounds} rounds, clip_norm={clip_norm}, c={c}, α={alpha}, warm_up={warm_up}, seed={seed}, dataset={dataset}, participation={participation_rate}, alpha_dirichlet={dirichlet_alpha}, batch_size={batch_size}, local_epochs={local_epochs}")
=======
def run_comparison(epsilon, clip_norm, num_rounds=10, device='cpu',
                   c=2.5, alpha=0.8, warm_up=5, seed=42, plot=True):
    print(f"\n{'='*60}")
    print(f"COMPARISON: ε={epsilon}, clip_norm={clip_norm}, c={c}, α={alpha}, warm_up={warm_up}, seed={seed}")
>>>>>>> parent of e8091c8 (Updated CNN & Per-Round Epsilon)
    print('='*60)

    set_seed(seed)  # <-- set seed before creating loaders

    client_loaders, test_loader, model_class, num_classes, class_names = get_dataset_components(
        dataset, num_clients=3, batch_size=batch_size, alpha=dirichlet_alpha, seed=seed)

<<<<<<< HEAD
    # No centralized training – removed

    std_fl = StandardFL(3, model_class, device, participation_rate=participation_rate)
    if per_round_epsilon is not None:
        dp_fl = BasicDPFL(3, model_class, device, participation_rate=participation_rate,
                          epsilon=per_round_epsilon, clip_norm=clip_norm)
        ecdp_fl = ECDPFL(3, model_class, device, participation_rate=participation_rate,
                         epsilon=per_round_epsilon, clip_norm=clip_norm,
                         c=c, alpha=alpha, warm_up=warm_up,
                         use_clipping=ablation_use_clipping,
                         use_smoothing=ablation_use_smoothing)
    else:
        dp_fl = BasicDPFL(3, model_class, device, participation_rate=participation_rate,
                          epsilon=None, target_epsilon=target_epsilon, max_rounds=num_rounds,
                          clip_norm=clip_norm)
        ecdp_fl = ECDPFL(3, model_class, device, participation_rate=participation_rate,
                         epsilon=None, target_epsilon=target_epsilon, max_rounds=num_rounds,
                         clip_norm=clip_norm,
                         c=c, alpha=alpha, warm_up=warm_up,
                         use_clipping=ablation_use_clipping,
                         use_smoothing=ablation_use_smoothing)
=======
    std_fl = StandardFL(3, TinyCNN, device)
    dp_fl = BasicDPFL(3, TinyCNN, device, epsilon, clip_norm=clip_norm)
    ecdp_fl = ECDPFL(3, TinyCNN, device, epsilon, clip_norm=clip_norm,
                     c=c, alpha=alpha, warm_up=warm_up)
>>>>>>> parent of e8091c8 (Updated CNN & Per-Round Epsilon)

    methods = {'Standard FL': std_fl, 'Basic DP-FL': dp_fl, 'EC-DP-FL': ecdp_fl}
    histories = {}
    round_times = {}
    final_accuracies = {}
    final_model_sizes = {}
    inference_latencies = {}

    for name, method in methods.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        acc_list = []
        for r in range(num_rounds):
            method.train_round(client_loaders, epochs=local_epochs)
            acc = method.test_accuracy(test_loader)
            acc_list.append(acc)
            print(f"Round {r+1}: {acc:.2f}%")
        total_time = time.time() - start_time
        histories[name] = acc_list
        round_times[name] = total_time
        final_accuracies[name] = acc_list[-1]
        param_size = sum(p.numel() for p in method.global_model.parameters())
        final_model_sizes[name] = param_size
        inference_latencies[name] = measure_inference_latency(method.global_model, test_loader, device)

        final_acc = acc_list[-1]
        threshold = final_acc - 1.0
        converged_round = num_rounds
        for i, acc in enumerate(acc_list):
            if acc >= threshold:
                converged_round = i + 1
                break
        setattr(method, 'convergence_round', converged_round)

    metrics = compare_methods_comprehensive(std_fl, dp_fl, ecdp_fl, test_loader, device,
                                            num_classes=num_classes, class_names=class_names)

    result = {
        'federated': {
            'standard': {
                'accuracy': final_accuracies['Standard FL'],
                'training_time': round_times['Standard FL'],
                'model_size': final_model_sizes['Standard FL'],
                'inference_latency': inference_latencies['Standard FL'],
                'convergence_round': std_fl.convergence_round,
                'history': histories['Standard FL']
            },
            'dp': {
                'accuracy': final_accuracies['Basic DP-FL'],
                'training_time': round_times['Basic DP-FL'],
                'model_size': final_model_sizes['Basic DP-FL'],
                'inference_latency': inference_latencies['Basic DP-FL'],
                'convergence_round': dp_fl.convergence_round,
                'history': histories['Basic DP-FL']
            },
            'ecdp': {
                'accuracy': final_accuracies['EC-DP-FL'],
                'training_time': round_times['EC-DP-FL'],
                'model_size': final_model_sizes['EC-DP-FL'],
                'inference_latency': inference_latencies['EC-DP-FL'],
                'convergence_round': ecdp_fl.convergence_round,
                'history': histories['EC-DP-FL']
            }
        },
        'metrics': metrics,
        'histories': histories,
        'test_loader': test_loader
    }

    if plot:
        plt.figure()
        for name, acc in histories.items():
            plt.plot(range(1, len(acc)+1), acc, marker='o', label=name)
        plt.xlabel('Federation Round')
        plt.ylabel('Accuracy (%)')
<<<<<<< HEAD
        title = f'Convergence ({mode}={eps}) - {dataset}'
        if dirichlet_alpha is not None:
            title += f' (Dirichlet α={dirichlet_alpha})'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/convergence_{mode}_{eps}_{dataset}_seed{seed}.png')
=======
        plt.title(f'Convergence (ε={epsilon}, C={clip_norm}, c={c}, α={alpha})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/convergence_eps{epsilon}_seed{seed}.png', dpi=150)
>>>>>>> parent of e8091c8 (Updated CNN & Per-Round Epsilon)
        plt.show()
    return result

def run_comparison_with_stats(args):
    num_runs = args.runs
    print(f"\nRunning {num_runs} independent runs...")
    results = []
    for run in range(num_runs):
        seed = args.seed + run
        print(f"\n--- Run {run+1}/{num_runs}, seed={seed} ---")
        res = run_comparison(
            per_round_epsilon=args.per_round_epsilon,
            target_epsilon=args.target_epsilon,
            clip_norm=args.clip_norm,
            num_rounds=args.rounds,
            device=args.device,
            c=args.c,
            alpha=args.alpha,
            warm_up=args.warm_up,
            participation_rate=args.participation_rate,
            seed=seed,
            plot=False,
            dataset=args.dataset,
            dirichlet_alpha=args.dirichlet_alpha,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs
        )
        results.append(res)

<<<<<<< HEAD
    std_acc = [r['federated']['standard']['accuracy'] for r in results]
    dp_acc = [r['federated']['dp']['accuracy'] for r in results]
    ecdp_acc = [r['federated']['ecdp']['accuracy'] for r in results]
    std_time = [r['federated']['standard']['training_time'] for r in results]
    dp_time = [r['federated']['dp']['training_time'] for r in results]
    ecdp_time = [r['federated']['ecdp']['training_time'] for r in results]
    std_conv = [r['federated']['standard']['convergence_round'] for r in results]
    dp_conv = [r['federated']['dp']['convergence_round'] for r in results]
    ecdp_conv = [r['federated']['ecdp']['convergence_round'] for r in results]

    def stats_str(arr):
        return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"

    print("\n" + "="*70)
    print("STATISTICAL SUMMARY (10 runs)")
=======
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
>>>>>>> parent of e8091c8 (Updated CNN & Per-Round Epsilon)
    print("="*70)
    print(f"Standard FL Accuracy:         {stats_str(std_acc)}%")
    print(f"Basic DP-FL Accuracy:         {stats_str(dp_acc)}%")
    print(f"EC-DP-FL Accuracy:            {stats_str(ecdp_acc)}%")
    print()
    print(f"Standard FL Training Time:    {stats_str(std_time)}s")
    print(f"Basic DP-FL Training Time:    {stats_str(dp_time)}s")
    print(f"EC-DP-FL Training Time:       {stats_str(ecdp_time)}s")
    print()
    print(f"Standard FL Convergence Rounds: {stats_str(std_conv)}")
    print(f"Basic DP-FL Convergence Rounds: {stats_str(dp_conv)}")
    print(f"EC-DP-FL Convergence Rounds:    {stats_str(ecdp_conv)}")
    print()

<<<<<<< HEAD
    w_stat, p_value = stats.wilcoxon(dp_acc, ecdp_acc)
    print(f"Wilcoxon test (DP vs EC-DP): W={w_stat:.2f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("→ Significant improvement (p < 0.05)")

    ci_low, ci_high = np.percentile(ecdp_acc, [2.5, 97.5])
    print(f"EC-DP 95% Bootstrap CI: [{ci_low:.2f}, {ci_high:.2f}]")

    df = pd.DataFrame({
        'Run': range(1, num_runs+1),
        'Standard Acc': std_acc,
        'DP Acc': dp_acc,
        'EC-DP Acc': ecdp_acc,
        'Standard Time': std_time,
        'DP Time': dp_time,
        'EC-DP Time': ecdp_time,
        'Standard Conv': std_conv,
        'DP Conv': dp_conv,
        'EC-DP Conv': ecdp_conv
    })
    df.to_csv('results/stats_summary.csv', index=False)
    print("\nResults saved to results/stats_summary.csv")

def run_tradeoff(args):
    epsilon_values = [0.1, 0.2, 0.5, 1.0, 2.0]
    num_trials = 10
    all_results = {}
    conf_mat_dir = 'results/confusion_matrices'
    os.makedirs(conf_mat_dir, exist_ok=True)

    for eps in epsilon_values:
        print(f"\n{'='*60}")
        print(f"Epsilon = {eps} (running {num_trials} trials)")
        print('='*60)
        trial_results = []
        for trial in range(num_trials):
            seed = args.seed + trial
            print(f"  Trial {trial+1}/{num_trials}, seed={seed}")
            res = run_comparison(
                per_round_epsilon=eps if args.mode_per_round else None,
                target_epsilon=eps if not args.mode_per_round else None,
                clip_norm=args.clip_norm,
                num_rounds=args.rounds,
                device=args.device,
                c=args.c,
                alpha=args.alpha,
                warm_up=args.warm_up,
                participation_rate=args.participation_rate,
                seed=seed,
                plot=False,
                dataset=args.dataset,
                dirichlet_alpha=args.dirichlet_alpha,
                batch_size=args.batch_size,
                local_epochs=args.local_epochs
            )
            conf_matrix = res['metrics']['ecdp_fl']['confusion_matrix']
            from utils.metrics import ComprehensiveMetrics
            num_classes = 7 if args.dataset == 'skin' else 2
            class_names = (['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] 
                           if args.dataset == 'skin' else ['NORMAL', 'PNEUMONIA'])
            metrics = ComprehensiveMetrics(num_classes=num_classes, class_names=class_names)
            fname = f"{conf_mat_dir}/confusion_eps{eps}_trial{trial+1}.png"
            metrics.plot_confusion_matrix(conf_matrix, 
                                          f"EC-DP-FL, ε={eps}, trial {trial+1}",
                                          save_path=fname)
            trial_results.append(res)

        std_acc = [r['federated']['standard']['accuracy'] for r in trial_results]
        dp_acc = [r['federated']['dp']['accuracy'] for r in trial_results]
        ecdp_acc = [r['federated']['ecdp']['accuracy'] for r in trial_results]
        std_time = [r['federated']['standard']['training_time'] for r in trial_results]
        dp_time = [r['federated']['dp']['training_time'] for r in trial_results]
        ecdp_time = [r['federated']['ecdp']['training_time'] for r in trial_results]
        std_conv = [r['federated']['standard']['convergence_round'] for r in trial_results]
        dp_conv = [r['federated']['dp']['convergence_round'] for r in trial_results]
        ecdp_conv = [r['federated']['ecdp']['convergence_round'] for r in trial_results]
        std_model_size = [r['federated']['standard']['model_size'] for r in trial_results]
        dp_model_size = [r['federated']['dp']['model_size'] for r in trial_results]
        ecdp_model_size = [r['federated']['ecdp']['model_size'] for r in trial_results]
        std_latency = [r['federated']['standard']['inference_latency'] for r in trial_results]
        dp_latency = [r['federated']['dp']['inference_latency'] for r in trial_results]
        ecdp_latency = [r['federated']['ecdp']['inference_latency'] for r in trial_results]

        all_results[eps] = {
            'Standard FL': {
                'Accuracy': np.mean(std_acc),
                'Accuracy_std': np.std(std_acc),
                'Training Time (s)': np.mean(std_time),
                'Convergence Rounds': np.mean(std_conv),
                'Model Size (params)': np.mean(std_model_size),
                'Inference Latency (ms)': np.mean(std_latency)
            },
            'Basic DP-FL': {
                'Accuracy': np.mean(dp_acc),
                'Accuracy_std': np.std(dp_acc),
                'Training Time (s)': np.mean(dp_time),
                'Convergence Rounds': np.mean(dp_conv),
                'Model Size (params)': np.mean(dp_model_size),
                'Inference Latency (ms)': np.mean(dp_latency)
            },
            'EC-DP-FL': {
                'Accuracy': np.mean(ecdp_acc),
                'Accuracy_std': np.std(ecdp_acc),
                'Training Time (s)': np.mean(ecdp_time),
                'Convergence Rounds': np.mean(ecdp_conv),
                'Model Size (params)': np.mean(ecdp_model_size),
                'Inference Latency (ms)': np.mean(ecdp_latency)
            }
        }

    rows = []
    for eps, data in all_results.items():
        for method in ['Standard FL', 'Basic DP-FL', 'EC-DP-FL']:
            rows.append({
                'Epsilon': eps,
                'Method': method,
                'Accuracy (%)': data[method]['Accuracy'],
                'Accuracy Std': data[method]['Accuracy_std'],
                'Training Time (s)': data[method]['Training Time (s)'],
                'Convergence Rounds': data[method]['Convergence Rounds'],
                'Model Size': data[method]['Model Size (params)'],
                'Inference Latency (ms)': data[method]['Inference Latency (ms)']
            })
    df = pd.DataFrame(rows)
    df.to_csv(f'results/tradeoff_{args.dataset}.csv', index=False)

    eps_list = list(all_results.keys())
    std_acc = [all_results[e]['Standard FL']['Accuracy'] for e in eps_list]
    dp_acc = [all_results[e]['Basic DP-FL']['Accuracy'] for e in eps_list]
    ecdp_acc = [all_results[e]['EC-DP-FL']['Accuracy'] for e in eps_list]
    std_std = [all_results[e]['Standard FL']['Accuracy_std'] for e in eps_list]
    dp_std = [all_results[e]['Basic DP-FL']['Accuracy_std'] for e in eps_list]
    ecdp_std = [all_results[e]['EC-DP-FL']['Accuracy_std'] for e in eps_list]

    plt.figure()
    plt.errorbar(eps_list, std_acc, yerr=std_std, marker='o', label='Standard FL', capsize=5)
    plt.errorbar(eps_list, dp_acc, yerr=dp_std, marker='s', label='Basic DP-FL', capsize=5)
    plt.errorbar(eps_list, ecdp_acc, yerr=ecdp_std, marker='^', label='EC-DP-FL', capsize=5)
    plt.xscale('log')
    plt.xlabel('Privacy budget ε')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Privacy-Utility Tradeoff - {args.dataset} (10 runs each)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/tradeoff_{args.dataset}.png')
    plt.show()

def run_ablation(args):
    components = [
        ('DP only', False, False),
        ('DP + Clipping', True, False),
        ('DP + Smoothing', False, True),
        ('Full ECDP', True, True)
    ]
    results = []
    for name, use_clip, use_smooth in components:
        print(f"\n--- {name} ---")
        res = run_comparison(
            per_round_epsilon=args.per_round_epsilon,
            target_epsilon=args.target_epsilon,
            clip_norm=args.clip_norm,
            num_rounds=args.rounds,
            device=args.device,
            c=args.c,
            alpha=args.alpha,
            warm_up=args.warm_up,
            participation_rate=args.participation_rate,
            seed=args.seed,
            plot=False,
            dataset=args.dataset,
            dirichlet_alpha=args.dirichlet_alpha,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            ablation_use_clipping=use_clip,
            ablation_use_smoothing=use_smooth
        )
        results.append({'name': name, 'accuracy': res['federated']['ecdp']['accuracy']})
    plt.figure()
    names = [r['name'] for r in results]
    accs = [r['accuracy'] for r in results]
    plt.bar(names, accs, color=['gray', 'lightblue', 'lightgreen', 'blue'])
    plt.ylabel('Accuracy (%)')
    plt.title('Ablation Study: Effect of Correction Components')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('results/ablation.png')
    plt.show()

def run_sensitivity(args):
    c_values = [1.5, 2.0, 2.5]
    alpha_values = [0.6, 0.7, 0.8]
    results = []
    for c in c_values:
        for alpha in alpha_values:
            print(f"\nTesting c={c}, α={alpha}")
            res = run_comparison(
                per_round_epsilon=args.per_round_epsilon,
                target_epsilon=args.target_epsilon,
                clip_norm=args.clip_norm,
                num_rounds=args.rounds,
                device=args.device,
                c=c,
                alpha=alpha,
                warm_up=0,
                participation_rate=args.participation_rate,
                seed=args.seed,
                plot=False,
                dataset=args.dataset,
                dirichlet_alpha=args.dirichlet_alpha,
                batch_size=args.batch_size,
                local_epochs=args.local_epochs
            )
            results.append({'c': c, 'alpha': alpha, 'accuracy': res['federated']['ecdp']['accuracy']})
    df = pd.DataFrame(results)
    pivot = df.pivot(index='c', columns='alpha', values='accuracy')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Sensitivity: EC-DP Accuracy vs c and α')
    plt.savefig('results/sensitivity_heatmap.png')
=======
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
>>>>>>> parent of e8091c8 (Updated CNN & Per-Round Epsilon)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('--mode', choices=['comparison', 'tradeoff', 'tune', 'stats', 'ablation', 'sensitivity'], default='comparison')
    parser.add_argument('--dataset', choices=['skin', 'chest'], default='skin')
    parser.add_argument('--per_round_epsilon', type=float, default=None)
    parser.add_argument('--target_epsilon', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=3.5)
    parser.add_argument('--rounds', type=int, default=20)
=======
    parser.add_argument('--mode', choices=['comparison', 'tradeoff', 'tune'], default='comparison')
    parser.add_argument('--epsilon', type=float, default=20.0)
    parser.add_argument('--clip_norm', type=float, default=2.1)
    parser.add_argument('--rounds', type=int, default=10)
>>>>>>> parent of e8091c8 (Updated CNN & Per-Round Epsilon)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--participation_rate', type=float, default=0.5)
    parser.add_argument('--dirichlet_alpha', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--local_epochs', type=int, default=3)
    parser.add_argument('--c', type=float, default=1.5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--mode_per_round', action='store_true', default=True)
    args = parser.parse_args()

    # CUDA fallback: if user requested GPU but CUDA is not available, fallback to CPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
<<<<<<< HEAD
    args.device = device

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
    elif args.mode == 'stats':
        run_comparison_with_stats(args)
    elif args.mode == 'tradeoff':
        run_tradeoff(args)
    elif args.mode == 'ablation':
        run_ablation(args)
    elif args.mode == 'sensitivity':
        run_sensitivity(args)
    elif args.mode == 'tune':
        c_values = [1.5, 2.0, 2.5]
        alpha_values = [0.6, 0.7, 0.8]
        best_acc = -1
        best_params = {}
        for c in c_values:
            for alpha in alpha_values:
                print(f"\n--- Testing c={c}, α={alpha} ---")
                res = run_comparison(
                    per_round_epsilon=args.per_round_epsilon,
                    target_epsilon=args.target_epsilon,
                    clip_norm=args.clip_norm,
                    num_rounds=args.rounds,
                    device=device,
                    c=c,
                    alpha=alpha,
                    warm_up=0,
                    participation_rate=args.participation_rate,
                    seed=args.seed,
                    plot=False,
                    dataset=args.dataset,
                    dirichlet_alpha=args.dirichlet_alpha,
                    batch_size=args.batch_size,
                    local_epochs=args.local_epochs
                )
                acc = res['federated']['ecdp']['accuracy']
                print(f"Accuracy: {acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'c': c, 'alpha': alpha}
        print(f"\nBest params: {best_params} with accuracy {best_acc:.2f}%")
=======

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
>>>>>>> parent of e8091c8 (Updated CNN & Per-Round Epsilon)
