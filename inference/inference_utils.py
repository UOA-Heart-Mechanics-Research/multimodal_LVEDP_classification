import os
import json
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

def plot_roc_curve(labels, predictions, name, mode='best', set='all'):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and saves it as a PNG file.

    Args:
        labels (array): True binary labels.
        predictions (array): Predicted probabilities for the positive class.
        name (str): Name of the directory to save the plot.
        mode (str, optional): Mode of the model ('best', 'all'). Defaults to 'best'.
        set (str, optional): Dataset type ('all', 'subset', 'indeterminate'). Defaults to 'all'.
    """
    # Compute ROC curve and AUC (Area Under the Curve)
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(15, 12))
    plt.plot(fpr, tpr, color='#A9C5A0', lw=10, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=5, linestyle='--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.xticks(fontsize=32)
    plt.ylim([0.0, 1.05])
    plt.yticks(fontsize=32)
    plt.xlabel('1-Specificity', fontsize=36)
    plt.ylabel('Sensitivity', fontsize=36)
    plt.title('ROC curve', fontsize=44)
    plt.legend(loc="lower right", fontsize=36)

    # Save the plot to the specified directory
    plt.savefig(os.path.join('runs', name, f'roc_curve_{mode}_{set}.png'))
    plt.close()


def compute_metrics(labels, predictions, test_loss, name, mode='best', set='all'):
    """
    Computes and saves classification metrics in a json file, and prints them to the console.

    Args:
        labels (array): True binary labels.
        predictions (array): Predicted binary labels.
        test_loss (float): Test loss value.
        name (str): Name of the directory to save the metrics.
        mode (str, optional): Mode of the model ('best', 'all'). Defaults to 'best'.
        set (str, optional): Dataset type ('all', 'subset', 'indeterminate'). Defaults to 'all'.
    """
    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # Compute classification metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0  # Recall or True Positive Rate
    specificity = tn / (tn + fp) if tn + fp > 0 else 0  # True Negative Rate
    precision = tp / (tp + fp) if tp + fp > 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if tn + fn > 0 else 0  # Negative Predictive Value

    # Save metrics to a JSON file
    metrics = {
        'Test Loss': test_loss,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Positive Predictive Value': precision,
        'Negative Predictive Value': npv
    }
    with open(os.path.join('runs', name, f'metrics_{mode}_{set}.json'), 'w') as f:
        json.dump(metrics, f)

    # Print metrics to the console
    print(f'Test Loss ({set}): {test_loss:.4f}')
    print(f'Accuracy ({set}): {accuracy:.4f}')
    print(f'Sensitivity ({set}): {sensitivity:.4f}')
    print(f'Specificity ({set}): {specificity:.4f}')
    print(f'Positive Predictive Value ({set}): {precision:.4f}')
    print(f'Negative Predictive Value ({set}): {npv:.4f}')