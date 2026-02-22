# utils/metrics.py - COMPLETE VERSION MATCHING MANUSCRIPT
"""
Comprehensive metrics implementation for Objectives 2.1.1-2.1.6
Matches manuscript specifications exactly
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensiveMetrics:
    """All performance metrics from Objectives 2.1"""
    
    def __init__(self, num_classes=7, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [
            'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
        ]
    
    def compute_all_metrics(self, model, test_loader, device):
        """
        Compute all metrics from Objectives 2.1.1-2.1.6
        
        Returns:
            dict with accuracy, precision, recall, f1, auc_roc, confusion_matrix
        """
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                
                # Get predictions
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Objective 2.1.1: Accuracy
        accuracy = accuracy_score(all_targets, all_preds) * 100
        
        # Objective 2.1.2: Precision
        precision = precision_score(all_targets, all_preds, 
                                   average='weighted', zero_division=0) * 100
        
        # Objective 2.1.3: Recall
        recall = recall_score(all_targets, all_preds, 
                            average='weighted', zero_division=0) * 100
        
        # Objective 2.1.4: F1-Score
        f1 = f1_score(all_targets, all_preds, 
                     average='weighted', zero_division=0) * 100
        
        # Objective 2.1.5: AUC-ROC (one-vs-rest for multiclass)
        try:
            auc_roc = roc_auc_score(all_targets, all_probs, 
                                   multi_class='ovr', average='weighted') * 100
        except ValueError:
            auc_roc = 0.0  # In case of single class in batch
        
        # Objective 2.1.6: Confusion Matrix
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def print_metrics_table(self, metrics_dict, method_name):
        """Print formatted metrics table"""
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
        """
        Objective 2.1.6: Confusion Matrix Visualization
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix: {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return plt.gcf()
    
    def generate_classification_report(self, targets, predictions):
        """Detailed per-class metrics"""
        return classification_report(
            targets, predictions,
            target_names=self.class_names,
            digits=3
        )
    
    def calculate_utility_metrics(self, std_metrics, dp_metrics, ecdp_metrics):
        """
        Calculate Eq. 9 and Eq. 10 from manuscript
        
        Eq. 9: Improvement = Accuracy_ECDP - Accuracy_BasicDP
        Eq. 10: Recovery Rate = Improvement / (Accuracy_StdFL - Accuracy_BasicDP) * 100%
        """
        # Eq. 9: Utility Improvement
        improvement = ecdp_metrics['accuracy'] - dp_metrics['accuracy']
        
        # Eq. 10: Utility Recovery Rate
        total_loss = std_metrics['accuracy'] - dp_metrics['accuracy']
        recovery_rate = (improvement / total_loss * 100) if total_loss > 0 else 0
        
        return {
            'improvement': improvement,
            'recovery_rate': recovery_rate,
            'dp_utility_loss': total_loss,
            'ecdp_utility_loss': std_metrics['accuracy'] - ecdp_metrics['accuracy']
        }

def compare_methods_comprehensive(std_fl, dp_fl, ecdp_fl, test_loader, device):
    """
    Comprehensive comparison matching Objectives 2.1 and 2.2
    """
    metrics = ComprehensiveMetrics()
    
    print("\n🔬 COMPUTING COMPREHENSIVE METRICS FOR ALL METHODS...")
    
    # Compute metrics for all three methods
    std_metrics = metrics.compute_all_metrics(std_fl.global_model, test_loader, device)
    dp_metrics = metrics.compute_all_metrics(dp_fl.global_model, test_loader, device)
    ecdp_metrics = metrics.compute_all_metrics(ecdp_fl.global_model, test_loader, device)
    
    # Print individual method results
    metrics.print_metrics_table(std_metrics, "Standard FL")
    metrics.print_metrics_table(dp_metrics, "Basic DP-FL")
    metrics.print_metrics_table(ecdp_metrics, "EC-DP-FL (Ours)")
    
    # Calculate utility metrics (Eq. 9 and 10)
    utility = metrics.calculate_utility_metrics(std_metrics, dp_metrics, ecdp_metrics)
    
    print(f"\n{'='*60}")
    print(f"🎯 UTILITY ANALYSIS (Eq. 9 & 10)")
    print(f"{'='*60}")
    print(f"DP Utility Loss:        {utility['dp_utility_loss']:>7.2f}%")
    print(f"EC-DP Utility Loss:     {utility['ecdp_utility_loss']:>7.2f}%")
    print(f"Improvement (Eq. 9):    {utility['improvement']:>7.2f}%")
    print(f"Recovery Rate (Eq. 10): {utility['recovery_rate']:>7.1f}%")
    print(f"{'='*60}\n")
    
    # Generate confusion matrices
    fig1 = metrics.plot_confusion_matrix(
        std_metrics['confusion_matrix'],
        "Standard FL",
        "results/confusion_matrix_std_fl.png"
    )
    
    fig2 = metrics.plot_confusion_matrix(
        dp_metrics['confusion_matrix'],
        "Basic DP-FL",
        "results/confusion_matrix_dp_fl.png"
    )
    
    fig3 = metrics.plot_confusion_matrix(
        ecdp_metrics['confusion_matrix'],
        "EC-DP-FL (Ours)",
        "results/confusion_matrix_ecdp_fl.png"
    )
    
    # Print classification reports
    print("\n📋 DETAILED CLASSIFICATION REPORT - EC-DP-FL:")
    print(metrics.generate_classification_report(
        ecdp_metrics['targets'],
        ecdp_metrics['predictions']
    ))
    
    return {
        'standard_fl': std_metrics,
        'dp_fl': dp_metrics,
        'ecdp_fl': ecdp_metrics,
        'utility': utility
    }