import argparse
import torch
import sys
import os
import random
import numpy as np
from scipy import stats   # for paired t-test

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import get_skin_cancer_dataloaders
from utils.chest_xray_loader import get_chest_xray_dataloaders
from utils.tb_uganda_loader import get_tb_uganda_dataloaders
from models.medium_cnn import MediumCNN
from models.chest_cnn import ChestCNN
from models.tb_cnn import TBCNN
from core.federated_learning import FederatedLearningBase as StandardFL
from core.dpfl import BasicDPFL, ECDPFL
from utils.metrics import ComprehensiveMetrics, SystemMetrics, compare_methods_comprehensive
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_comparison(per_round_epsilon=None, target_epsilon=None, clip_norm=2.3, num_rounds=10,
                   device='cpu', c=2.5, alpha=0.8, seed=42, plot=True, iid=True,
                   dirichlet_alpha=0.5, dataset='skin', num_clients=10):
    print(f"\n{'='*60}")
    if per_round_epsilon is not None:
        mode = "per‑round ε"
        eps = per_round_epsilon
    else:
        mode = "total ε"
        eps = target_epsilon
    dist_type = "IID" if iid else f"non-IID (α={dirichlet_alpha})"
    print(f"COMPARISON: {mode}={eps} over {num_rounds} rounds, dataset={dataset}, "
          f"clients={num_clients}, clip_norm={clip_norm}, c={c}, α={alpha}, seed={seed}, "
          f"distribution={dist_type}")
    print('='*60)

    set_seed(seed)

    # ---------- Dataset and model selection ----------
    if dataset == 'skin':
        client_loaders, test_loader = get_skin_cancer_dataloaders(
            num_clients=num_clients, batch_size=64,
            iid=iid, dirichlet_alpha=dirichlet_alpha, seed=seed)
        model_class = MediumCNN
        num_classes = 7
    elif dataset == 'chest':
        client_loaders, test_loader = get_chest_xray_dataloaders(
            num_clients=num_clients, batch_size=64,
            iid=iid, dirichlet_alpha=dirichlet_alpha, seed=seed)
        model_class = ChestCNN
        num_classes = 2
    elif dataset == 'tb_uganda':
        client_loaders, test_loader = get_tb_uganda_dataloaders(
            num_clients=num_clients, batch_size=32,
            iid=iid, dirichlet_alpha=dirichlet_alpha, seed=seed)
        model_class = TBCNN
        num_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # ---------- Initialize methods ----------
    std_fl = StandardFL(num_clients, lambda: model_class(num_classes=num_classes), device)
    
    if per_round_epsilon is not None:
        dp_fl = BasicDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                          epsilon=per_round_epsilon, clip_norm=clip_norm)
        ecdp_fl = ECDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                         epsilon=per_round_epsilon, clip_norm=clip_norm,
                         c=c, alpha=alpha)
    else:
        dp_fl = BasicDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                          epsilon=None, target_epsilon=target_epsilon, max_rounds=num_rounds,
                          clip_norm=clip_norm)
        ecdp_fl = ECDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                         epsilon=None, target_epsilon=target_epsilon, max_rounds=num_rounds,
                         clip_norm=clip_norm, c=c, alpha=alpha)

    methods = {'Standard FL': std_fl, 'Basic DP-FL': dp_fl, 'EC-DP-FL': ecdp_fl}
    histories = {}
    round_times = {}

    for name, method in methods.items():
        print(f"\n--- Training {name} ---")
        acc_list = []
        for r in range(num_rounds):
            method.train_round(client_loaders, epochs=2)
            acc = method.test_accuracy(test_loader)
            acc_list.append(acc)
            print(f"Round {r+1}: {acc:.2f}% (time: {method.round_times[-1]:.2f}s)")
        histories[name] = acc_list
        round_times[name] = method.round_times.copy()
        total_time = sum(method.round_times)
        avg_time = total_time / len(method.round_times)
        print(f"\n{name} total training time: {total_time:.2f}s, avg per round: {avg_time:.2f}s")

    metrics = compare_methods_comprehensive(std_fl, dp_fl, ecdp_fl, test_loader, device)

    if plot:
        plt.figure(figsize=(10,6))
        for name, acc in histories.items():
            plt.plot(range(1, len(acc)+1), acc, marker='o', label=name)
        plt.xlabel('Federation Round')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Convergence ({mode}={eps}, {dist_type}, dataset={dataset})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/convergence_{mode}_{eps}_{dataset}_seed{seed}_'
                    f'{"IID" if iid else f"nonIID_alpha{dirichlet_alpha}"}.png', dpi=150)
        plt.show()

    return metrics, histories, test_loader, round_times

def run_validation(per_round_epsilon=None, target_epsilon=None, clip_norm=2.3, num_rounds=10,
                   num_trials=10, device='cpu', c=2.5, alpha=0.8, base_seed=42,
                   iid=True, dirichlet_alpha=0.5, dataset='skin', num_clients=10,
                   use_evc=True, use_ags=True):
    """
    Run validation with explicit control over EVC and AGS components.
    
    Args:
        use_evc: Enable Extreme Value Clipping (default: True)
        use_ags: Enable Adaptive Gradient Smoothing (default: True)
    """
    print("\n" + "="*70)
    print("STATISTICAL VALIDATION (10 independent runs)")
    if per_round_epsilon is not None:
        mode = "per‑round ε"
        eps = per_round_epsilon
    else:
        mode = "total ε"
        eps = target_epsilon
    dist_type = "IID" if iid else f"non-IID (α={dirichlet_alpha})"
    print(f"Configuration: {mode}={eps}, rounds={num_rounds}, dataset={dataset}, clients={num_clients}, "
          f"clip_norm={clip_norm}, c={c}, α={alpha}, distribution={dist_type}")
    print(f"Components: EVC={use_evc}, AGS={use_ags}")
    print("="*70)

    all_std_metrics = []
    all_dp_metrics = []
    all_ecdp_metrics = []
    all_std_history = []
    all_dp_history = []
    all_ecdp_history = []
    all_std_round_times = []
    all_dp_round_times = []
    all_ecdp_round_times = []

    for trial in range(num_trials):
        seed = base_seed + trial
        print(f"\n--- Trial {trial+1}/{num_trials} (seed={seed}) ---")
        set_seed(seed)

        # ---------- Dataset and model selection ----------
        if dataset == 'skin':
            client_loaders, test_loader = get_skin_cancer_dataloaders(
                num_clients=num_clients, batch_size=64,
                iid=iid, dirichlet_alpha=dirichlet_alpha, seed=seed)
            model_class = MediumCNN
            num_classes = 7
            class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        elif dataset == 'chest':
            client_loaders, test_loader = get_chest_xray_dataloaders(
                num_clients=num_clients, batch_size=64,
                iid=iid, dirichlet_alpha=dirichlet_alpha, seed=seed)
            model_class = ChestCNN
            num_classes = 2
            class_names = ['NORMAL', 'PNEUMONIA']
        elif dataset == 'tb_uganda':
            client_loaders, test_loader = get_tb_uganda_dataloaders(
                num_clients=num_clients, batch_size=32,
                iid=iid, dirichlet_alpha=dirichlet_alpha, seed=seed)
            model_class = TBCNN
            num_classes = 2
            class_names = ['NORMAL', 'TB']
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # ---------- Initialize methods ----------
        std_fl = StandardFL(num_clients, lambda: model_class(num_classes=num_classes), device)
        
        if per_round_epsilon is not None:
            dp_fl = BasicDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                              epsilon=per_round_epsilon, clip_norm=clip_norm)
            ecdp_fl = ECDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                             epsilon=per_round_epsilon, clip_norm=clip_norm,
                             c=c, alpha=alpha, use_evc=use_evc, use_ags=use_ags)
        else:
            dp_fl = BasicDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                              epsilon=None, target_epsilon=target_epsilon,
                              max_rounds=num_rounds, clip_norm=clip_norm)
            ecdp_fl = ECDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                             epsilon=None, target_epsilon=target_epsilon,
                             max_rounds=num_rounds, clip_norm=clip_norm,
                             c=c, alpha=alpha, use_evc=use_evc, use_ags=use_ags)

        # Train all methods
        std_acc_history = []
        for r in range(num_rounds):
            std_fl.train_round(client_loaders, epochs=2)
            acc = std_fl.test_accuracy(test_loader)
            std_acc_history.append(acc)
        all_std_history.append(std_acc_history)
        all_std_round_times.append(std_fl.round_times)

        dp_acc_history = []
        for r in range(num_rounds):
            dp_fl.train_round(client_loaders, epochs=2)
            acc = dp_fl.test_accuracy(test_loader)
            dp_acc_history.append(acc)
        all_dp_history.append(dp_acc_history)
        all_dp_round_times.append(dp_fl.round_times)

        ecdp_acc_history = []
        for r in range(num_rounds):
            ecdp_fl.train_round(client_loaders, epochs=2)
            acc = ecdp_fl.test_accuracy(test_loader)
            ecdp_acc_history.append(acc)
        all_ecdp_history.append(ecdp_acc_history)
        all_ecdp_round_times.append(ecdp_fl.round_times)

        # Compute final metrics
        metrics_calc = ComprehensiveMetrics(num_classes=num_classes, class_names=class_names)
        std_met = metrics_calc.compute_all_metrics(std_fl.global_model, test_loader, device)
        dp_met  = metrics_calc.compute_all_metrics(dp_fl.global_model, test_loader, device)
        ecdp_met = metrics_calc.compute_all_metrics(ecdp_fl.global_model, test_loader, device)

        all_std_metrics.append(std_met)
        all_dp_metrics.append(dp_met)
        all_ecdp_metrics.append(ecdp_met)

        # Print trial results (as percentages)
        print(f"  Std FL: {std_met['accuracy']:.2f}% | DP-FL: {dp_met['accuracy']:.2f}% | "
              f"EC-DP-FL: {ecdp_met['accuracy']:.2f}%")

    # ---------- Aggregate results ----------
    def aggregate(metric_list, key):
        values = [m[key] for m in metric_list]
        return np.mean(values), np.std(values)

    print("\n" + "="*70)
    print(f"📊 FINAL RESULTS (Mean ± Std over {num_trials} trials)")
    print("="*70)

    for name, metric_list in [("Standard FL", all_std_metrics),
                              ("Basic DP-FL", all_dp_metrics),
                              ("EC-DP-FL", all_ecdp_metrics)]:
        print(f"\n{name}:")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            mean_val, std_val = aggregate(metric_list, metric)
            print(f"  {metric.capitalize():10s}: {mean_val:6.2f} ± {std_val:5.2f}%")

    # ---------- Per‑trial comparison (ECDP vs Basic DP) ----------
    print("\n" + "="*70)
    print("PER-TRIAL ANALYSIS (EC-DP-FL vs Basic DP-FL)")
    print("="*70)
    dp_accs = [m['accuracy'] for m in all_dp_metrics]
    ecdp_accs = [m['accuracy'] for m in all_ecdp_metrics]
    
    ecdp_wins = sum(1 for e, d in zip(ecdp_accs, dp_accs) if e > d)
    print(f"EC-DP-FL wins: {ecdp_wins}/{num_trials} ({ecdp_wins/num_trials*100:.1f}%)")
    
    try:
        t_stat, p_value = stats.ttest_rel(ecdp_accs, dp_accs)
        print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05 and np.mean(ecdp_accs) > np.mean(dp_accs):
            print("Significant improvement: YES ✓")
        else:
            print("Significant improvement: NO")
    except Exception as e:
        print(f"Could not compute t-test: {e}")

    # Save results to file
    os.makedirs('results', exist_ok=True)
    with open('results/validation_results.txt', 'w') as f:
        f.write(f"Validation results for {mode}={eps}, {num_trials} trials, dataset={dataset}, distribution={dist_type}\n")
        f.write(f"Components: EVC={use_evc}, AGS={use_ags}\n")
        f.write(f"Standard FL accuracy: {aggregate(all_std_metrics, 'accuracy')[0]:.2f}±{aggregate(all_std_metrics, 'accuracy')[1]:.2f}%\n")
        f.write(f"Basic DP-FL accuracy: {aggregate(all_dp_metrics, 'accuracy')[0]:.2f}±{aggregate(all_dp_metrics, 'accuracy')[1]:.2f}%\n")
        f.write(f"EC-DP-FL accuracy: {aggregate(all_ecdp_metrics, 'accuracy')[0]:.2f}±{aggregate(all_ecdp_metrics, 'accuracy')[1]:.2f}%\n")
        f.write(f"EC-DP-FL wins: {ecdp_wins}/{num_trials}\n")
    print("\n✅ Validation complete. Results saved to results/validation_results.txt")

    return {
        'std':  all_std_metrics,
        'dp':   all_dp_metrics,
        'ecdp': all_ecdp_metrics,
    }

def run_ablation(per_round_epsilon=None, target_epsilon=None, clip_norm=2.3,
                 num_rounds=10, device='cpu', base_seed=42, c=2.5, alpha=0.8,
                 iid=True, dirichlet_alpha=0.5, dataset='skin', num_clients=10):
    print("\n" + "="*70)
    print("ABLATION STUDY")
    if per_round_epsilon is not None:
        mode = "per‑round ε"
        eps = per_round_epsilon
    else:
        mode = "total ε"
        eps = target_epsilon
    dist_type = "IID" if iid else f"non-IID (α={dirichlet_alpha})"
    print(f"Configuration: {mode}={eps}, rounds={num_rounds}, dataset={dataset}, clients={num_clients}, "
          f"clip_norm={clip_norm}, c={c}, α={alpha}, distribution={dist_type}")
    print("="*70)

    set_seed(base_seed)
    if dataset == 'skin':
        client_loaders, test_loader = get_skin_cancer_dataloaders(
            num_clients=num_clients, batch_size=64,
            iid=iid, dirichlet_alpha=dirichlet_alpha, seed=base_seed)
        model_class = MediumCNN
        num_classes = 7
    elif dataset == 'chest':
        client_loaders, test_loader = get_chest_xray_dataloaders(
            num_clients=num_clients, batch_size=64,
            iid=iid, dirichlet_alpha=dirichlet_alpha, seed=base_seed)
        model_class = ChestCNN
        num_classes = 2
    elif dataset == 'tb_uganda':
        client_loaders, test_loader = get_tb_uganda_dataloaders(
            num_clients=num_clients, batch_size=32,
            iid=iid, dirichlet_alpha=dirichlet_alpha, seed=base_seed)
        model_class = TBCNN
        num_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    metrics_calc = ComprehensiveMetrics(num_classes=num_classes)

    configs = [
        ("Standard FL", False, False, False),
        ("DP only", True, False, False),
        ("DP + EVC", True, True, False),
        ("DP + AGS", True, False, True),
    ]

    results = {}
    round_times = {}
    for name, use_dp, use_evc, use_ags in configs:
        print(f"\n--- Running: {name} ---")
        if not use_dp:
            model = StandardFL(num_clients, lambda: model_class(num_classes=num_classes), device)
        else:
            if per_round_epsilon is not None:
                model = ECDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                               epsilon=per_round_epsilon, clip_norm=clip_norm,
                               use_evc=use_evc, use_ags=use_ags, c=c, alpha=alpha)
            else:
                model = ECDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                               epsilon=None, target_epsilon=target_epsilon, max_rounds=num_rounds,
                               clip_norm=clip_norm, use_evc=use_evc, use_ags=use_ags, c=c, alpha=alpha)
        for r in range(num_rounds):
            model.train_round(client_loaders, epochs=2)
        met = metrics_calc.compute_all_metrics(model.global_model, test_loader, device)
        results[name] = met
        round_times[name] = model.round_times.copy()
        total_time = sum(model.round_times)
        avg_time = total_time / len(model.round_times)
        print(f"Total training time: {total_time:.2f}s, avg per round: {avg_time:.2f}s")
        metrics_calc.print_metrics_table(met, name)

    os.makedirs('results', exist_ok=True)
    with open('results/ablation_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Ablation Study Results (dataset={dataset}, distribution={dist_type})\n")
        f.write("="*60 + "\n")
        for name, met in results.items():
            total_time = sum(round_times[name])
            f.write(f"{name}:\n")
            f.write(f"  Accuracy:  {met['accuracy']:.2f}%\n")
            f.write(f"  Precision: {met['precision']:.2f}%\n")
            f.write(f"  Recall:    {met['recall']:.2f}%\n")
            f.write(f"  F1-Score:  {met['f1_score']:.2f}%\n")
            f.write(f"  AUC-ROC:   {met['auc_roc']:.2f}%\n")
            f.write(f"  Total time: {total_time:.2f}s\n\n")
    print("\n✅ Ablation results saved to results/ablation_results.txt")

def tune_correction_params(per_round_epsilon=None, target_epsilon=None, clip_norm=2.3,
                           num_rounds=10, device='cpu', c_values=[1.5, 2.0, 2.5],
                           alpha_values=[0.6, 0.7, 0.8], seed=42, iid=True,
                           dirichlet_alpha=0.5, dataset='skin', num_clients=10):
    best_acc = -1
    best_params = {}
    for c in c_values:
        for alpha in alpha_values:
            print(f"\n--- Testing c={c}, α={alpha} ---")
            _, histories, _, _ = run_comparison(per_round_epsilon, target_epsilon, clip_norm,
                                                num_rounds, device, c, alpha, seed=seed,
                                                plot=False, iid=iid,
                                                dirichlet_alpha=dirichlet_alpha,
                                                dataset=dataset, num_clients=num_clients)
            acc = histories['EC-DP-FL'][-1]
            print(f"Final EC-DP-FL accuracy: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_params = {'c': c, 'alpha': alpha}
    print(f"\nBest params: {best_params} with accuracy {best_acc:.2f}%")
    return best_params

def run_tradeoff(epsilon_values, clip_norm, num_rounds=10, device='cpu', base_seed=42,
                 mode='per_round', iid=True, dirichlet_alpha=0.5, dataset='skin', num_clients=10):
    print("\n" + "="*70)
    print(f"PRIVACY‑UTILITY TRADEOFF ANALYSIS ({mode} ε)")
    dist_type = "IID" if iid else f"non-IID (α={dirichlet_alpha})"
    print(f"Dataset: {dataset}, Distribution: {dist_type}, Clients: {num_clients}")
    print("="*70)

    basic_accs = []
    ecdp_accs = []

    for eps in epsilon_values:
        print(f"\n--- {mode} ε = {eps} ---")
        set_seed(base_seed)
        if dataset == 'skin':
            client_loaders, test_loader = get_skin_cancer_dataloaders(
                num_clients=num_clients, batch_size=64,
                iid=iid, dirichlet_alpha=dirichlet_alpha, seed=base_seed)
            model_class = MediumCNN
            num_classes = 7
        elif dataset == 'chest':
            client_loaders, test_loader = get_chest_xray_dataloaders(
                num_clients=num_clients, batch_size=64,
                iid=iid, dirichlet_alpha=dirichlet_alpha, seed=base_seed)
            model_class = ChestCNN
            num_classes = 2
        elif dataset == 'tb_uganda':
            client_loaders, test_loader = get_tb_uganda_dataloaders(
                num_clients=num_clients, batch_size=32,
                iid=iid, dirichlet_alpha=dirichlet_alpha, seed=base_seed)
            model_class = TBCNN
            num_classes = 2
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if eps <= 0.5:
            c, alpha = 1.5, 0.6
        else:
            c, alpha = 2.5, 0.8

        if mode == 'per_round':
            dp = BasicDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                           epsilon=eps, clip_norm=clip_norm)
            ec = ECDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                        epsilon=eps, clip_norm=clip_norm, c=c, alpha=alpha)
        else:
            dp = BasicDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                           epsilon=None, target_epsilon=eps, max_rounds=num_rounds, clip_norm=clip_norm)
            ec = ECDPFL(num_clients, lambda: model_class(num_classes=num_classes), device,
                        epsilon=None, target_epsilon=eps, max_rounds=num_rounds,
                        clip_norm=clip_norm, c=c, alpha=alpha)

        for r in range(num_rounds):
            dp.train_round(client_loaders, epochs=2)
            ec.train_round(client_loaders, epochs=2)

        basic_acc = dp.test_accuracy(test_loader)
        ecdp_acc = ec.test_accuracy(test_loader)
        basic_accs.append(basic_acc)
        ecdp_accs.append(ecdp_acc)
        print(f"  Basic DP: {basic_acc:.2f}%")
        print(f"  EC-DP:    {ecdp_acc:.2f}%")

    # Get Standard FL baseline
    set_seed(base_seed)
    if dataset == 'skin':
        client_loaders, test_loader = get_skin_cancer_dataloaders(
            num_clients=num_clients, batch_size=64,
            iid=iid, dirichlet_alpha=dirichlet_alpha, seed=base_seed)
        model_class = MediumCNN
        num_classes = 7
    elif dataset == 'chest':
        client_loaders, test_loader = get_chest_xray_dataloaders(
            num_clients=num_clients, batch_size=64,
            iid=iid, dirichlet_alpha=dirichlet_alpha, seed=base_seed)
        model_class = ChestCNN
        num_classes = 2
    elif dataset == 'tb_uganda':
        client_loaders, test_loader = get_tb_uganda_dataloaders(
            num_clients=num_clients, batch_size=32,
            iid=iid, dirichlet_alpha=dirichlet_alpha, seed=base_seed)
        model_class = TBCNN
        num_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    std_fl = StandardFL(num_clients, lambda: model_class(num_classes=num_classes), device)
    for r in range(num_rounds):
        std_fl.train_round(client_loaders, epochs=2)
    std_acc = std_fl.test_accuracy(test_loader)

    plt.figure(figsize=(10,6))
    plt.plot(epsilon_values, basic_accs, marker='o', label='Basic DP-FL')
    plt.plot(epsilon_values, ecdp_accs, marker='s', label='EC-DP-FL')
    plt.axhline(y=std_acc, color='g', linestyle='--', label='Standard FL')
    plt.xscale('log')
    plt.xlabel(f'Privacy budget ε ({mode})')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Privacy‑Utility Tradeoff ({mode} ε, {dataset}, {dist_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/tradeoff_{mode}_{dataset}_'
                f'{"IID" if iid else f"nonIID_alpha{dirichlet_alpha}"}.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['comparison', 'tradeoff', 'tune', 'validation', 'ablation'],
                        default='comparison')
    parser.add_argument('--dataset', choices=['skin', 'chest', 'tb_uganda'], default='skin',
                        help='Dataset to use (skin=HAM10000, chest=pneumonia, tb_uganda=Uganda TB X-ray)')
    parser.add_argument('--per_round_epsilon', type=float, default=None)
    parser.add_argument('--target_epsilon', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=2.3)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--c', type=float, default=2.5)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--non_iid', action='store_true', help='Use non-IID data distribution (Dirichlet)')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                        help='Dirichlet concentration (lower = more heterogeneous)')
    parser.add_argument('--clients', type=int, default=10,
                        help='Number of simulated clients (default: 10)')
    parser.add_argument('--use_evc', action='store_true', default=True,
                        help='Enable Extreme Value Clipping (default: True)')
    parser.add_argument('--use_ags', action='store_true', default=True,
                        help='Enable Adaptive Gradient Smoothing (default: True)')
    parser.add_argument('--no_evc', dest='use_evc', action='store_false',
                        help='Disable Extreme Value Clipping')
    parser.add_argument('--no_ags', dest='use_ags', action='store_false',
                        help='Disable Adaptive Gradient Smoothing')
    args = parser.parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.mode in ['comparison', 'tune', 'validation', 'ablation']:
        assert (args.per_round_epsilon is not None) ^ (args.target_epsilon is not None), \
            "Exactly one of --per_round_epsilon or --target_epsilon must be provided."

    iid = not args.non_iid
    dirichlet_alpha = args.dirichlet_alpha

    if args.mode == 'comparison':
        run_comparison(args.per_round_epsilon, args.target_epsilon, args.clip_norm, args.rounds, device,
                       args.c, args.alpha, seed=args.seed, iid=iid,
                       dirichlet_alpha=dirichlet_alpha, dataset=args.dataset, num_clients=args.clients)
    elif args.mode == 'tradeoff':
        epsilon_list = [0.1, 1.0, 2.0]
        run_tradeoff(epsilon_list, args.clip_norm, args.rounds, device, base_seed=args.seed,
                     mode='per_round', iid=iid, dirichlet_alpha=dirichlet_alpha,
                     dataset=args.dataset, num_clients=args.clients)
    elif args.mode == 'tune':
        tune_correction_params(args.per_round_epsilon, args.target_epsilon, args.clip_norm, args.rounds,
                               device, seed=args.seed, iid=iid,
                               dirichlet_alpha=dirichlet_alpha, dataset=args.dataset, num_clients=args.clients)
    elif args.mode == 'validation':
        run_validation(args.per_round_epsilon, args.target_epsilon, args.clip_norm, args.rounds,
                       args.trials, device, args.c, args.alpha, base_seed=args.seed,
                       iid=iid, dirichlet_alpha=dirichlet_alpha,
                       dataset=args.dataset, num_clients=args.clients,
                       use_evc=args.use_evc, use_ags=args.use_ags)
    elif args.mode == 'ablation':
        run_ablation(args.per_round_epsilon, args.target_epsilon, args.clip_norm, args.rounds,
                     device, base_seed=args.seed, c=args.c, alpha=args.alpha,
                     iid=iid, dirichlet_alpha=dirichlet_alpha,
                     dataset=args.dataset, num_clients=args.clients)