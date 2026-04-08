import torch
import os
import numpy as np
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class ComprehensiveMetrics:
    def __init__(self, num_classes=None, class_names=None):
        """
        num_classes : if None, inferred from data.
        class_names : list of human-readable names indexed by label integer.
                      Length must equal the actual number of unique labels seen.
        """
        self.num_classes = num_classes
        self.class_names = class_names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unique_labels(self, targets, predictions):
        """Sorted list of every label that appears in targets OR predictions."""
        return sorted(set(np.unique(targets)) | set(np.unique(predictions)))

    def _label_names(self, unique_labels):
        """
        Map integer labels to display names.
        Falls back to 'Class N' if class_names is None or the wrong length.
        """
        if (
            self.class_names is not None
            and len(self.class_names) == max(unique_labels) + 1
        ):
            return [self.class_names[i] for i in unique_labels]
        return [f"Class {i}" for i in unique_labels]

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def compute_all_metrics(self, model, test_loader, device):
        model.eval()
        all_preds   = []
        all_targets = []
        all_probs   = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                probs   = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds   = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs   = np.array(all_probs)

        if self.num_classes is None:
            self.num_classes = len(np.unique(all_targets))

        accuracy  = accuracy_score(all_targets, all_preds) * 100
        precision = precision_score(all_targets, all_preds,
                                    average='weighted', zero_division=0) * 100
        recall    = recall_score(all_targets, all_preds,
                                 average='weighted', zero_division=0) * 100
        f1        = f1_score(all_targets, all_preds,
                             average='weighted', zero_division=0) * 100

        try:
            auc_roc = roc_auc_score(
                all_targets, all_probs,
                multi_class='ovr', average='weighted'
            ) * 100
        except ValueError:
            auc_roc = 0.0

        conf_matrix = confusion_matrix(all_targets, all_preds)

        return {
            'accuracy':         accuracy,
            'precision':        precision,
            'recall':           recall,
            'f1_score':         f1,
            'auc_roc':          auc_roc,
            'confusion_matrix': conf_matrix,
            'predictions':      all_preds,
            'targets':          all_targets,
        }

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_metrics_table(self, metrics_dict, method_name):
        print(f"\n{'='*60}")
        print(f"📊 COMPREHENSIVE METRICS: {method_name}")
        print(f"{'='*60}")
        print(f"Accuracy:     {metrics_dict['accuracy']:>7.2f}%")
        print(f"Precision:    {metrics_dict['precision']:>7.2f}%")
        print(f"Recall:       {metrics_dict['recall']:>7.2f}%")
        print(f"F1-Score:     {metrics_dict['f1_score']:>7.2f}%")
        print(f"AUC-ROC:      {metrics_dict['auc_roc']:>7.2f}%")
        print(f"{'='*60}\n")

    def plot_confusion_matrix(self, conf_matrix, title, save_path=None):
        n = conf_matrix.shape[0]
        label_names = self._label_names(list(range(n)))

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names)
        plt.title(f'Confusion Matrix: {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return plt.gcf()

    def generate_classification_report(self, targets, predictions):
        """
        Build a sklearn classification report that always passes the correct
        labels list, avoiding the 'Number of classes does not match
        target_names' error that occurred when using a chest (2-class) dataset
        with skin (7-class) class_names.
        """
        unique_labels = self._unique_labels(targets, predictions)
        label_names   = self._label_names(unique_labels)

        return classification_report(
            targets, predictions,
            labels=unique_labels,
            target_names=label_names,
            digits=3,
            zero_division=0,
        )

    def calculate_utility_metrics(self, std_metrics, dp_metrics, ecdp_metrics):
        improvement  = ecdp_metrics['accuracy'] - dp_metrics['accuracy']
        total_loss   = std_metrics['accuracy']  - dp_metrics['accuracy']
        recovery_rate = (improvement / total_loss * 100) if total_loss > 0 else 0
        return {
            'improvement':       improvement,
            'recovery_rate':     recovery_rate,
            'dp_utility_loss':   total_loss,
            'ecdp_utility_loss': std_metrics['accuracy'] - ecdp_metrics['accuracy'],
        }


# ---------------------------------------------------------------------------
# System-level helpers (unchanged)
# ---------------------------------------------------------------------------

class SystemMetrics:
    @staticmethod
    def compute_training_time(round_times):
        total_time = sum(round_times)
        avg_time   = np.mean(round_times)
        std_time   = np.std(round_times)
        return total_time, avg_time, std_time

    @staticmethod
    def compute_convergence_rate(accuracy_history, target_ratio=0.95):
        if not accuracy_history:
            return 0
        final_acc  = accuracy_history[-1]
        target_acc = target_ratio * final_acc
        for i, acc in enumerate(accuracy_history):
            if acc >= target_acc:
                return i + 1
        return len(accuracy_history)

    @staticmethod
    def measure_inference_latency(model, sample_input, device,
                                  num_warmup=10, num_runs=100):
        model.eval()
        sample_input = sample_input.to(device)
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(sample_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        return (elapsed / num_runs) * 1000

    @staticmethod
    def get_model_size(model):
        param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Top-level comparison helper
# ---------------------------------------------------------------------------

def compare_methods_comprehensive(std_fl, dp_fl, ecdp_fl, test_loader, device):
    # Infer num_classes from the first batch
    sample_batch = next(iter(test_loader))
    num_classes  = int(sample_batch[1].max().item()) + 1

    # Pick sensible class_names based on the dataset size
    if num_classes == 2:
        class_names = ['NORMAL', 'PNEUMONIA']
    elif num_classes == 7:
        # HAM10000 label order (label 0–6)
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    else:
        class_names = None

    metrics = ComprehensiveMetrics(num_classes=num_classes, class_names=class_names)

    std_metrics  = metrics.compute_all_metrics(std_fl.global_model,  test_loader, device)
    dp_metrics   = metrics.compute_all_metrics(dp_fl.global_model,   test_loader, device)
    ecdp_metrics = metrics.compute_all_metrics(ecdp_fl.global_model, test_loader, device)

    metrics.print_metrics_table(std_metrics,  "Standard FL")
    metrics.print_metrics_table(dp_metrics,   "Basic DP-FL")
    metrics.print_metrics_table(ecdp_metrics, "EC-DP-FL (Ours)")

    utility = metrics.calculate_utility_metrics(std_metrics, dp_metrics, ecdp_metrics)
    print(f"\n{'='*60}")
    print(f"🎯 UTILITY ANALYSIS (Eq. 9 & 10)")
    print(f"{'='*60}")
    print(f"DP Utility Loss:        {utility['dp_utility_loss']:>7.2f}%")
    print(f"EC-DP Utility Loss:     {utility['ecdp_utility_loss']:>7.2f}%")
    print(f"Improvement (Eq. 9):    {utility['improvement']:>7.2f}%")
    print(f"Recovery Rate (Eq. 10): {utility['recovery_rate']:>7.1f}%")
    print(f"{'='*60}\n")

    # Confusion matrices
    os.makedirs('results', exist_ok=True)
    metrics.plot_confusion_matrix(
        std_metrics['confusion_matrix'],  "Standard FL",
        "results/confusion_matrix_std_fl.png")
    metrics.plot_confusion_matrix(
        dp_metrics['confusion_matrix'],   "Basic DP-FL",
        "results/confusion_matrix_dp_fl.png")
    metrics.plot_confusion_matrix(
        ecdp_metrics['confusion_matrix'], "EC-DP-FL (Ours)",
        "results/confusion_matrix_ecdp_fl.png")

    print("\n📋 DETAILED CLASSIFICATION REPORT - EC-DP-FL:")
    print(metrics.generate_classification_report(
        ecdp_metrics['targets'], ecdp_metrics['predictions']))

    return {
        'standard_fl': std_metrics,
        'dp_fl':       dp_metrics,
        'ecdp_fl':     ecdp_metrics,
        'utility':     utility,
    }