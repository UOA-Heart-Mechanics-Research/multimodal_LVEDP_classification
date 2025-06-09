import os
import sys
import torch
from models.multimodal_model import MultimodalModel
from train.losses import WeightedBCELoss
from ops import args_parser
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils import set_global_seed
from config.config import load_config, save_config_and_args, CONFIG
from logger import Logger
from data.data_utils import get_dataloaders
from train.trainer import train
from inference.evaluate import evaluate


def main(args, data_path, mode='normal'):
    """
    Main function to handle training and evaluation (inference) of the model.

    Args:
        args: Parsed command-line arguments.
        data_path: Path to the dataset folder.
        mode: Mode of operation, either 'normal' or 'cross_val' for cross-validation.
    """
    # Determine the number of folds for cross-validation, default to 1 if normal mode
    if mode == 'cross_val':
        nb_folds = len(os.listdir(os.path.join(data_path, 'folds')))
    else:
        nb_folds = 1

    for fold in range(nb_folds):
        # Set the data path for the current fold
        if mode == 'cross_val':
            fold_data_path = os.path.join(data_path, 'folds', f'fold_{fold}')
        else:
            fold_data_path = data_path

        # Reset global seed before each fold training for reproducibility
        seed = CONFIG.training.seed
        set_global_seed(seed)

        # Get data loaders for training, validation, and testing
        train_loader, val_loader, test_loader = get_dataloaders(args, fold_data_path, data_path, seed=seed)

        # Initialize the model
        model = MultimodalModel(CONFIG.model.resnet_out_dim, CONFIG.model.tgcn_out_dim,
                                CONFIG.model.nb_clinical_features, CONFIG.model.alignment_dim, 
                                args.modalities).to(device='cuda' if torch.cuda.is_available() else 'cpu')

        # Define loss functions, optimizer, and learning rate scheduler
        pos_weight = CONFIG.training.positive_weight 
        null_weight = CONFIG.training.null_weight
        criterion = WeightedBCELoss(pos_weight, null_weight)
        val_criterion = WeightedBCELoss(pos_weight, null_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.training.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG.training.T_0, 
                                                T_mult=CONFIG.training.T_mult, eta_min=CONFIG.training.eta_min)

        # Train the model if training is enabled
        if args.train:
            if mode == 'cross_val':
                print(f"\nTraining fold {fold}")
                name = os.path.join(args.name, f'fold_{fold}')
            else:
                name = args.name

            train(name, model, train_loader, val_loader, criterion, val_criterion, optimizer, scheduler, args.num_epochs)

        # Evaluate the model on the test set
        if mode == 'cross_val':
            print(f"\nEvaluating fold {fold}")
            name = os.path.join(args.name, f'fold_{fold}')
        else:
            name = args.name

        # Load the best and last checkpoints for evaluation
        model.load_state_dict(torch.load(os.path.join('runs', name, 'checkpoints', 'model_best.pth')))
        evaluate(name, model, test_loader, criterion, 'best')
        model.load_state_dict(torch.load(os.path.join('runs', name, 'checkpoints', 'model_last.pth')))
        evaluate(name, model, test_loader, criterion, 'last')

    # Perform cross-validation evaluation if in cross_val mode
    if mode == 'cross_val':
        print(f"\nEvaluating cross fold")
        for model_type in ['best', 'last']:
            models = []
            for fold in range(nb_folds):
                name = os.path.join(args.name, f'fold_{fold}')
                model = MultimodalModel(CONFIG.model.resnet_out_dim, CONFIG.model.tgcn_out_dim,
                                CONFIG.model.nb_clinical_features, CONFIG.model.alignment_dim, 
                                args.modalities).to(device='cuda' if torch.cuda.is_available() else 'cpu')
                model.load_state_dict(torch.load(os.path.join('runs', name, 'checkpoints', f'model_{model_type}.pth')))
                models.append(model)
            evaluate(args.name, models, test_loader, criterion, model_type, cross_val=True)


if __name__ == "__main__":
    # Parse command-line arguments
    args = args_parser()

    # Load configuration settings from the specified config file
    load_config(args.config)

    # Save the configuration settings and command-line arguments to a YAML file if training
    if args.train:
        yaml_file = os.path.join('runs', args.name, 'config.yaml')
        save_config_and_args(args, yaml_file)

    # Set the global random seed for reproducibility
    set_global_seed(CONFIG.training.seed)

    # Redirect stdout to a log file for saving terminal outputs
    log_file = os.path.join('runs', args.name, 'terminal_logs.txt')
    sys.stdout = Logger(log_file)

    # Define the path to the dataset folder
    data_path = os.path.join(args.input_dir)

    # Run the main function in cross-validation mode if specified, otherwise normal mode
    if args.cross_val:
        main(args, data_path, mode='cross_val')
    else:
        main(args, data_path, mode='normal')

    # Restore the original stdout after execution
    sys.stdout = sys.__stdout__







