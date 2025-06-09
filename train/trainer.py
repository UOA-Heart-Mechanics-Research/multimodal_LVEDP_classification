import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def train(name, model, train_loader, val_loader, criterion, val_criterion, optimizer, scheduler, num_epochs):
    """
    Train a multimodal PyTorch model.

    Args:
        name (str): Name of the training run (used for saving logs and checkpoints).
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (callable): Loss function for training.
        val_criterion (callable): Loss function for validation.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        None
    """
    # Create directory for saving models if it doesn't exist
    save_path = os.path.join('runs', name, 'checkpoints')
    os.makedirs(save_path, exist_ok=True)
    best_val_loss = float('inf')

    # Initialize TensorBoard writer for logging
    writer = SummaryWriter(os.path.join('runs', name, 'logs'))

    for epoch in range(num_epochs):
        # Set epoch for reproducibility in data shuffling
        train_loader.dataset.set_epoch(epoch)

        # Set model to training mode
        model.train()
        train_loss = 0

        # Log learning rate to TensorBoard
        writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Learning rate: {optimizer.param_groups[0]['lr']}")

        # Progress bar for training
        train_progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] | Training")
        optimizer.zero_grad()

        for step, (x, case) in enumerate(train_progress_bar):
            # Forward pass
            outputs, _ = model(x)

            # Compute loss
            loss = criterion(outputs, x['label'])

            # Backward pass and optimization
            loss.backward()

            # Compute and log gradient norm
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm(2).item() ** 2
            total_norm = total_norm ** 0.5  # L2 norm
            writer.add_scalar("Gradients/Norm", total_norm, epoch * len(train_loader) + step)

            optimizer.step()
            optimizer.zero_grad()

            # Accumulate training loss
            train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        # Average training loss and log it
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Validation step
        model.eval()
        val_loss = 0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] | Validation")

        all_labels = []
        all_predictions_binary = []

        with torch.no_grad():
            for x, case in val_progress_bar:
                # Forward pass
                outputs, _ = model(x)

                # Compute validation loss
                loss = val_criterion(outputs, x['label'])
                val_loss += loss.item()

                # Apply sigmoid activation and threshold for binary predictions
                outputs = torch.sigmoid(outputs)
                predicted_binary = (outputs > 0.5).float()

                # Collect labels and predictions for metrics
                all_labels.extend(x['label'].cpu().numpy())
                all_predictions_binary.extend(predicted_binary.cpu().numpy())
                val_progress_bar.set_postfix(loss=loss.item())

        # Average validation loss and log it
        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # Log train and validation loss together
        writer.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, epoch)

        # Compute performance metrics
        all_labels = np.array(all_labels)
        all_predictions_binary = np.array(all_predictions_binary)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions_binary).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if tp + fn > 0 else 0  # True Positive Rate
        specificity = tn / (tn + fp) if tn + fp > 0 else 0  # True Negative Rate
        precision = tp / (tp + fp) if tp + fp > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if tn + fn > 0 else 0  # Negative Predictive Value

        # Log metrics to TensorBoard
        writer.add_scalar('Metrics/accuracy', accuracy, epoch)
        writer.add_scalar('Metrics/sensitivity', sensitivity, epoch)
        writer.add_scalar('Metrics/specificity', specificity, epoch)
        writer.add_scalar('Metrics/precision', precision, epoch)
        writer.add_scalar('Metrics/npv', npv, epoch)
        writer.add_scalars('Metrics', {'accuracy': accuracy, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'precision': precision, 'npv': npv}, epoch)
        writer.add_scalars('Losses and Metrics', {'train_loss': train_loss, 'val_loss': val_loss,
                                                  'accuracy': accuracy, 'sensitivity': sensitivity,
                                                  'specificity': specificity, 'precision': precision, 'npv': npv}, epoch)

        # Save the last model checkpoint
        torch.save(model, os.path.join(save_path, 'model_architecture.pth'))  # Save model architecture
        torch.save(model.state_dict(), os.path.join(save_path, 'model_last.pth'))  # Save model weights

        # Step the learning rate scheduler
        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        writer.flush()

        # Save the best model checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'model_best.pth'))

    print('Best epoch:', best_epoch)
    writer.close()
