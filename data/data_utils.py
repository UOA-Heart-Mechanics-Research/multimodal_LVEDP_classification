import os
from torch.utils.data import DataLoader
from .dataset import MultiDataset
from .transforms import augment_video, augment_mesh, augment_clinical

def get_dataloaders(args, train_val_data_path, test_data_path, seed):
        """
        Creates and returns DataLoader objects for training, validation, and test datasets.

        Args:
                args: An object containing configuration parameters such as batch_size and transform_prob.
                train_val_data_path (str): Path to the directory containing training and validation data.
                test_data_path (str): Path to the directory containing test data.
                seed (int): Random seed for reproducibility.

        Returns:
                tuple: A tuple containing three DataLoader objects (train_loader, val_loader, test_loader).
        """
        # Create the training dataset with data augmentation
        multi_dataset_train = MultiDataset(
                os.path.join(train_val_data_path, 'train'),
                seed=seed,
                transform_prob=args.transform_prob,
                transform_video=augment_video,
                transform_mesh=augment_mesh,
                transform_clinical=augment_clinical
        )
        # DataLoader for the training dataset
        train_loader = DataLoader(
                multi_dataset_train,
                batch_size=args.batch_size,
                shuffle=True,  # Shuffle the data for training
                num_workers=0  # Number of subprocesses for data loading
        )

        # Create the validation dataset without data augmentation
        multi_dataset_val = MultiDataset(
                os.path.join(train_val_data_path, 'val'),
                seed=seed
        )
        # DataLoader for the validation dataset
        val_loader = DataLoader(
                multi_dataset_val,
                batch_size=1,  # Batch size of 1 for validation
                shuffle=False,  # No shuffling for validation
                num_workers=0
        )

        # Create the test dataset without data augmentation
        multi_dataset_test = MultiDataset(
                os.path.join(test_data_path, 'test'),
                seed=seed
        )
        # DataLoader for the test dataset
        test_loader = DataLoader(
                multi_dataset_test,
                batch_size=1,  # Batch size of 1 for testing
                shuffle=False,  # No shuffling for testing
                num_workers=0
        )

        return train_loader, val_loader, test_loader