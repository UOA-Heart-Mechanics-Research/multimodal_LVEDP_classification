import argparse
from datetime import datetime

def args_parser():
    """
    Parses command-line arguments for configuring the model training and evaluation process.

    Returns:
        argparse.Namespace: Parsed arguments with their values.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=str(datetime.now()).replace(' ', '_'), help="Name of the model")
    parser.add_argument('--num_epochs', type=int, default=35, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--cross_val', action='store_true', help='Perform cross validation')
    parser.add_argument('--transform_prob', type=float, default=0.5, help='Probability of applying data transformations')
    parser.add_argument('--modalities', nargs='+', type=str, default=['A2C', 'A4C', 'mesh', 'clinical'], help='Modalities to use')
    parser.add_argument('--input_dir', type=str, default='', help='Input directory')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file')

    # Parse and return the arguments
    args = parser.parse_args()

    return args
