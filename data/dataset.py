import os
import torch
import random
import h5py
import numpy as np
from torch.utils.data import Dataset
from .transforms import normalise_image, standardise_mesh, standardise_vector

class MultiDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing multi-modal data from HDF5 files.
    """

    def __init__(self, root_dir, transform_prob=0, transform_video=None, transform_mesh=None, transform_clinical=None, seed=None):
        """
        Initialize the dataset with the root directory and optional transformations.

        Args:
            root_dir (str): Directory with all the HDF5 files.
            transform_prob (float): Probability of applying transformations.
            transform_video (callable, optional): Transformation function for video data.
            transform_mesh (callable, optional): Transformation function for mesh data.
            transform_clinical (callable, optional): Transformation function for clinical data.
            seed (int, optional): Seed for reproducibility.
        """
        self.seed = seed
        self.root_dir = root_dir
        self.transform_prob = transform_prob
        self.transform_video = transform_video
        self.transform_mesh = transform_mesh
        self.transform_clinical = transform_clinical
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch = 0
        print(f"Using device: {self.device}")

    def set_epoch(self, epoch):
        """
        Set the current epoch for the dataset.

        Args:
            epoch (int): Current epoch number.
        """
        self.epoch = epoch

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        """
        Retrieve a sample and its corresponding label by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the sample data and its label.
            str: The case file identifier.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Save global random states to ensure reproducibility
        random_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()

        # Load the HDF5 file corresponding to the given index
        case_file = os.listdir(self.root_dir)[idx]
        case_seed = int(case_file.split('.')[0][2:])  # Extract seed from filename
        case_random = random.Random(self.seed + case_seed if self.seed is not None else case_seed)

        h5_file_path = os.path.join(self.root_dir, case_file)
        with h5py.File(h5_file_path, 'r') as hf:
            # Load A2C and A4C video data and normalise them
            A2C_series, A4C_series, mesh_series = hf.get('A2C_images'), hf.get('A4C_images'), hf.get('meshes')
            A2C_items, A4C_items, mesh_items = list(A2C_series.items()), list(A4C_series.items()), list(mesh_series.items())
            choice_A2C = case_random.choice(A2C_items)[0]
            A2C_video = torch.tensor(np.array(A2C_series.get(choice_A2C))).permute(3, 0, 1, 2).repeat(3, 1, 1, 1).to(torch.float32).to(self.device)
            A2C_video = normalise_image(A2C_video)

            choice_A4C = case_random.choice(A4C_items)[0]
            A4C_video = torch.tensor(np.array(A4C_series.get(choice_A4C))).permute(3, 0, 1, 2).repeat(3, 1, 1, 1).to(torch.float32).to(self.device)
            A4C_video = normalise_image(A4C_video)

            # Load mesh data and standardise it
            choice_mesh = case_random.choice(mesh_items)[0]
            mesh = np.array(mesh_series.get(choice_mesh))
            mesh = torch.tensor(standardise_mesh(mesh)).to(torch.float32).to(self.device)

            # Load clinical measurements and scale them
            measurements = torch.tensor(hf.get('measurements')).to(torch.float32)
            measurements = standardise_vector(measurements).to(self.device)

            # Load graph data (edge index and attributes)
            edge_index = torch.tensor(np.array(hf.get('edge_index'))).type(torch.LongTensor).to(self.device)
            edge_attr = torch.tensor(np.array(hf.get('edge_attr'))).to(torch.float32).to(self.device)

            # Load label
            label = torch.tensor(hf.get('label')).to(torch.float32).to(self.device)

            # Create a sample dictionary
            sample = {'A2C_video': A2C_video, 'A4C_video': A4C_video, 'mesh': mesh, 'measurements': measurements,
                      'edge_index': edge_index, 'edge_attr': edge_attr, 'label': label}

        # Apply transformations if specified
        if self.transform_video or self.transform_mesh or self.transform_clinical:
            random.seed(self.seed + case_seed + self.epoch if self.seed is not None else case_seed + self.epoch)
            do_transform = random.random() < self.transform_prob
        else:
            do_transform = False

        if do_transform:
            if self.transform_video:
                sample['A2C_video'] = self.transform_video(sample['A2C_video'], self.seed + case_seed + self.epoch if self.seed is not None else case_seed + self.epoch)
                sample['A4C_video'] = self.transform_video(sample['A4C_video'], self.seed + case_seed + self.epoch if self.seed is not None else case_seed + self.epoch)
                # Debugging: Check if video values exceed the expected range
                if sample['A2C_video'].max().cpu().numpy() > 1 or sample['A4C_video'].max().cpu().numpy() > 1:
                    print(f"Max value in A2C video: {sample['A2C_video'].max().cpu().numpy()}")
                    print(f"Max value in A4C video: {sample['A4C_video'].max().cpu().numpy()}")
            if self.transform_mesh:
                sample['mesh'] = self.transform_mesh(sample['mesh'], self.seed + case_seed + self.epoch if self.seed is not None else case_seed + self.epoch)
            if self.transform_clinical:
                sample['measurements'] = self.transform_clinical(sample['measurements'], self.seed + case_seed + self.epoch if self.seed is not None else case_seed + self.epoch)

        # Restore global random states
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

        # Return the sample and the case file identifier (without the extension)
        case_file = case_file.split('.')[0]
        return sample, case_file
