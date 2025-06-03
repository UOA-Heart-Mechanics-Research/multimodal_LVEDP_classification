import random
import numpy as np
import torch

def set_global_seed(seed):
    """
    Set the global random seed for reproducibility across random, numpy, and PyTorch.
    
    Args:
        seed (int): The seed value to set for all random number generators.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy's random number generator
    np.random.seed(seed)
    
    # Set seed for PyTorch's random number generator
    torch.manual_seed(seed)
    
    # If CUDA is available, set seed for CUDA-specific random number generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic behavior in CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
