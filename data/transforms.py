import torch
import numpy as np
import random
import torchvision.transforms.functional as F
from config.config import CONFIG


""" Procrustes function and data transformation utilities for preprocessing."""

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    source: https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy

    A port of MATLAB's `procrustes` function to Numpy.

    Courtesy of Dr. Debbie Zhao.

    Perform Procrustes analysis to determine the linear transformation
    (translation, reflection, rotation, scaling) that best aligns Y to X.

    Args:
        X (ndarray): Target matrix of shape (n, m).
        Y (ndarray): Input matrix of shape (n, my).
        scaling (bool): Whether to include scaling in the transformation.
        reflection (str): 'best', True, or False to control reflection.

    Returns:
        tuple: Residual sum of squared errors (d), transformed Y (Z), and
               transformation parameters (tform).
    """
    # Center the matrices
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY

    # Normalise the matrices
    normX = np.sqrt((X0 ** 2).sum())
    normY = np.sqrt((Y0 ** 2).sum())
    X0 /= normX
    Y0 /= normY

    # Compute the optimal rotation matrix
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Handle reflection if specified
    if reflection != 'best' and (np.linalg.det(T) < 0) != reflection:
        V[:, -1] *= -1
        s[-1] *= -1
        T = np.dot(V, U.T)

    traceTA = s.sum()

    # Compute scaling and transformed coordinates
    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + (Y0 ** 2).sum() / (X0 ** 2).sum() - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # Compute translation
    c = muX - b * np.dot(muY, T)

    # Return transformation parameters
    tform = {'rotation': T, 'scale': b, 'translation': c}
    return d, Z, tform


def normalise_image(image):
    """
    Normalise an image to have values between 0 and 1.

    Args:
        image (torch.Tensor): Input image as a tensor.

    Returns:
        torch.Tensor: Normalised image as a tensor.
    """
    return image / 255.0


def standardise_mesh(mesh):
    """
    Standardise a 3D mesh by aligning it to a reference shape and scaling.

    Args:
        mesh (ndarray): Input mesh of shape (frames, points, 3).

    Returns:
        ndarray: Standardised mesh.
    """
    reference_shape = np.load('resources/reference_mesh.npy')
    _, _, tform = procrustes(reference_shape, mesh[0], scaling=False, reflection=False)

    # Align each frame
    aligned_frames = [
        tform['scale'] * np.dot(frame_points, tform['rotation']) + tform['translation']
        for frame_points in mesh
    ]

    # Center and scale the mesh
    mean_vals = [CONFIG.data.mesh_stats.mean_val_x, CONFIG.data.mesh_stats.mean_val_y, CONFIG.data.mesh_stats.mean_val_z]
    max_dim = CONFIG.data.mesh_stats.max_dim
    standardised_mesh = (mesh - mean_vals) / max_dim

    return standardised_mesh


def standardise_vector(vector):
    """
    Standardise a clinical vector by subtracting the mean and dividing by twice the standard deviation.

    Args:
        vector (Tensor): Clinical vector.

    Returns:
        Tensor: Scaled clinical vector.
    """
    means = torch.tensor([
        0, CONFIG.data.clinical_stats.mean_age, CONFIG.data.clinical_stats.mean_weight,
        CONFIG.data.clinical_stats.mean_height, CONFIG.data.clinical_stats.mean_sbp,
        CONFIG.data.clinical_stats.mean_dbp
    ])
    stds = torch.tensor([
        1, CONFIG.data.clinical_stats.std_age, CONFIG.data.clinical_stats.std_weight,
        CONFIG.data.clinical_stats.std_height, CONFIG.data.clinical_stats.std_sbp,
        CONFIG.data.clinical_stats.std_dbp
    ])
    return (vector - means) / (2 * stds)


def augment_video(video, seed):
    """
    Apply consistent affine transformations and color jitter to a video.

    Args:
        video (Tensor): Input video of shape (channel, frames, width, height).
        seed (int): Random seed for reproducibility.

    Returns:
        Tensor: Augmented video.
    """
    random.seed(seed)
    channel, frame_count, width, height = video.shape

    # Set transformation parameters
    angle = random.uniform(-30, 30)
    translate = (random.uniform(-10, 10), random.uniform(-10, 10))
    scale = random.uniform(0.9, 1.1)
    brightness, contrast, gamma = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)
    horizontal_flip, vertical_flip = random.random() > 0.7, random.random() > 0.7

    # Apply transformations to each frame
    transformed_frames = []
    for i in range(frame_count):
        frame = video[:, i, :, :]
        transformed_frame = F.affine(frame, angle=angle, translate=translate, scale=scale, shear=0)
        transformed_frame = F.adjust_brightness(transformed_frame, brightness)
        transformed_frame = F.adjust_contrast(transformed_frame, contrast)
        transformed_frame = F.adjust_gamma(transformed_frame, gamma)
        if horizontal_flip:
            transformed_frame = F.hflip(transformed_frame)
        if vertical_flip:
            transformed_frame = F.vflip(transformed_frame)
        transformed_frames.append(transformed_frame)

    return torch.stack(transformed_frames, dim=1)


def augment_mesh(mesh, seed):
    """
    Apply small uniform scaling to a 3D mesh.

    Args:
        mesh (Tensor): Input mesh of shape (frames, nodes, features).
        seed (int): Random seed for reproducibility.

    Returns:
        Tensor: Augmented mesh.
    """
    random.seed(seed)
    scale_factor = np.random.uniform(0.9, 1.1)
    return mesh * scale_factor


def augment_clinical(clinical_vector, seed):
    """
    Add Gaussian noise to continuous clinical measurements.

    Args:
        clinical_vector (Tensor): Clinical vector.
        seed (int): Random seed for reproducibility.

    Returns:
        Tensor: Augmented clinical vector.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    noise = torch.normal(0, 0.05, clinical_vector.shape).to(clinical_vector.device)
    clinical_vector[0] = clinical_vector[0]  # Preserve binary value (gender)
    return clinical_vector + noise
