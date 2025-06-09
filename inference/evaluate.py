import torch
import numpy as np
from .inference_utils import compute_metrics, plot_roc_curve
from config.config import CONFIG

def evaluate(name, models, test_loader, criterion, mode='best', cross_val=False):
    """
    Evaluate the performance of one or more models on a test dataset.

    Args:
        name (str): Name of the evaluation run (used for saving results).
        models (list or torch.nn.Module): A single model or a list of models to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used for evaluation.
        mode (str): Mode of evaluation ('best' or 'last').
        cross_val (bool): Whether to perform cross-validation (averaging predictions across models).

    Returns:
        None
    """
    # Ensure models is a list
    if not isinstance(models, list):
        models = [models]
    num_folds = len(models)

    # Set all models to evaluation mode
    for model in models:
        model.eval()

    # Initialize variables to track metrics
    test_loss = 0
    all_labels = []
    all_predictions = []
    all_predictions_binary = []

    subset_test_loss = 0 # subset = all-indeterminate
    subset_labels = []
    subset_predictions = []
    subset_predictions_binary = []

    indeterminate_test_loss = 0
    indeterminate_labels = []
    indeterminate_predictions = []
    indeterminate_predictions_binary = []

    # Get the list of indeterminate cases from the configuration
    indeterminate_cases = CONFIG.test.indeterminate_cases 

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for x, case in test_loader:
            # Initialize outputs tensor
            outputs = torch.tensor([[0.0]]).to(device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Aggregate predictions and losses across models
            for model in models:
                outputs_torch, _ = model(x)
                loss = criterion(outputs_torch, x['label'])
                if cross_val:
                    test_loss += (loss.item() / num_folds)
                    outputs_torch = torch.sigmoid(outputs_torch)
                    outputs += (outputs_torch / num_folds)
                else:
                    test_loss += loss.item()
                    outputs = torch.sigmoid(outputs_torch)
            
            # Convert predictions to binary (threshold = 0.5)
            predicted_binary = (outputs > 0.5).float()
            predicted = outputs.float()

            # Collect all predictions and labels
            all_labels.extend(x['label'].cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_predictions_binary.extend(predicted_binary.cpu().numpy())

            # Separate metrics for indeterminate and subset cases
            if indeterminate_cases:
                if case[0] not in indeterminate_cases:
                    subset_test_loss += loss.item()
                    subset_labels.extend(x['label'].cpu().numpy())
                    subset_predictions.extend(predicted.cpu().numpy())
                    subset_predictions_binary.extend(predicted_binary.cpu().numpy())
                else:
                    indeterminate_test_loss += loss.item()
                    indeterminate_labels.extend(x['label'].cpu().numpy())
                    indeterminate_predictions.extend(predicted.cpu().numpy())
                    indeterminate_predictions_binary.extend(predicted_binary.cpu().numpy())

    # Convert lists to numpy arrays for metric calculations
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_predictions_binary = np.array(all_predictions_binary)
    
    if indeterminate_cases:
        subset_labels = np.array(subset_labels)
        subset_predictions = np.array(subset_predictions)
        subset_predictions_binary = np.array(subset_predictions_binary)

        indeterminate_labels = np.array(indeterminate_labels)
        indeterminate_predictions = np.array(indeterminate_predictions)
        indeterminate_predictions_binary = np.array(indeterminate_predictions_binary)

    # Compute and log metrics for all cases
    test_loss /= len(test_loader)
    compute_metrics(all_labels, all_predictions_binary, test_loss, name, mode)
    
    if indeterminate_cases:
        # Compute and log metrics for subset and indeterminate cases
        subset_test_loss /= (len(test_loader) - len(indeterminate_cases))
        indeterminate_test_loss /= len(indeterminate_cases)
        compute_metrics(subset_labels, subset_predictions_binary, subset_test_loss, name, mode, set='subset')
        compute_metrics(indeterminate_labels, indeterminate_predictions_binary, indeterminate_test_loss, name, mode, set='indeterminate')

    # Compute and plot ROC curves
    plot_roc_curve(all_labels, all_predictions, name, mode)
    if indeterminate_cases:
        plot_roc_curve(subset_labels, subset_predictions, name, mode, set='subset')
        plot_roc_curve(indeterminate_labels, indeterminate_predictions, name, mode, set='indeterminate')
