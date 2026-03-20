import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensiveMetrics:
    def __init__(self, num_classes=7, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    def compute_all_metrics(self, model, test_loader, device):
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_targets, all_preds) * 100
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
        try:
            auc_roc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted') * 100
        except ValueError:
            auc_roc = 0.0
        conf_matrix = confusion_matrix(all_targets, all_preds)

        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1, 'auc_roc': auc_roc, 'confusion_matrix': conf_matrix,
            'predictions': all_preds, 'targets': all_targets
        }

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
        plt.figure(figsize=(10,8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix: {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return plt.gcf()

    def generate_classification_report(self, targets, predictions):
        return classification_report(targets, predictions, target_names=self.class_names, digits=3)

    def calculate_utility_metrics(self, std_metrics, dp_metrics, ecdp_metrics):
        improvement = ecdp_metrics['accuracy'] - dp_metrics['accuracy']
        total_loss = std_metrics['accuracy'] - dp_metrics['accuracy']
        recovery_rate = (improvement / total_loss * 100) if total_loss > 0 else 0
        return {
            'improvement': improvement,
            'recovery_rate': recovery_rate,
            'dp_utility_loss': total_loss,
            'ecdp_utility_loss': std_metrics['accuracy'] - ecdp_metrics['accuracy']
        }

def compare_methods_comprehensive(std_fl, dp_fl, ecdp_fl, test_loader, device, num_classes=7, class_names=None):
    metrics = ComprehensiveMetrics(num_classes=num_classes, class_names=class_names)
    print("\n🔬 COMPUTING COMPREHENSIVE METRICS FOR ALL METHODS...")
    std_metrics = metrics.compute_all_metrics(std_fl.global_model, test_loader, device)
    dp_metrics = metrics.compute_all_metrics(dp_fl.global_model, test_loader, device)
    ecdp_metrics = metrics.compute_all_metrics(ecdp_fl.global_model, test_loader, device)

    metrics.print_metrics_table(std_metrics, "Standard FL")
    metrics.print_metrics_table(dp_metrics, "Basic DP-FL")
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

    os.makedirs('results', exist_ok=True)
    metrics.plot_confusion_matrix(std_metrics['confusion_matrix'], "Standard FL", "results/confusion_matrix_std_fl.png")
    metrics.plot_confusion_matrix(dp_metrics['confusion_matrix'], "Basic DP-FL", "results/confusion_matrix_dp_fl.png")
    metrics.plot_confusion_matrix(ecdp_metrics['confusion_matrix'], "EC-DP-FL (Ours)", "results/confusion_matrix_ecdp_fl.png")

    print("\n📋 DETAILED CLASSIFICATION REPORT - EC-DP-FL:")
    print(metrics.generate_classification_report(ecdp_metrics['targets'], ecdp_metrics['predictions']))

    return {
        'standard_fl': std_metrics,
        'dp_fl': dp_metrics,
        'ecdp_fl': ecdp_metrics,
        'utility': utility
    }