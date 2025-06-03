import os
import torch
from torch import nn
from .modules import TemporalMeshGCN, SimpleFCN, VideoEncoder


os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resnet')

class MultimodalModel(torch.nn.Module):
    """
    A PyTorch model for multimodal data classification. This model combines features from multiple modalities
    (videos, meshes, and clinical measurements) and performs classification using a fully connected network.

    Args:
        resnet_out_dim (int): Output dimension of the ResNet3D video encoder.
        tgcn_out_dim (int): Output dimension of the Temporal Graph Convolutional Network (TGCN).
        measurement_dim (int): Input dimension of clinical measurements.
        aligned_dim (int): Dimension to which all modalities are aligned.
        modalities (list): List of modalities to include in the model (e.g., ['A2C', 'A4C', 'mesh', 'clinical']).
    """

    def __init__(self, resnet_out_dim, tgcn_out_dim, measurement_dim, aligned_dim, modalities):
        super(MultimodalModel, self).__init__()

        self.modalities = modalities

        # Video encoders for A2C and A4C views using ResNet3D
        self.video_encoder_A2C = VideoEncoder(resnet_out_dim)
        self.video_encoder_A4C = VideoEncoder(resnet_out_dim)

        # Graph encoder for mesh data using Temporal Graph Convolutional Network (TGCN)
        self.graph_encoder_mesh = TemporalMeshGCN(3, tgcn_out_dim)

        # Fully connected network for clinical measurements
        self.measurement_encoder = SimpleFCN(measurement_dim, aligned_dim)

        # Alignment layers to align features from different modalities to the same dimension
        self.align_A2C = SimpleFCN(128, aligned_dim)
        self.align_A4C = SimpleFCN(128, aligned_dim)
        self.align_mesh = SimpleFCN(tgcn_out_dim, aligned_dim)

        # Classifier to combine aligned features and perform final classification
        self.classifier = nn.Sequential(
            nn.Linear(aligned_dim * len(modalities), 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Args:
            x (dict): Input data containing the following keys:
                - 'A2C_video': Tensor for A2C video data.
                - 'A4C_video': Tensor for A4C video data.
                - 'mesh': Tensor for mesh data.
                - 'measurements': Tensor for clinical measurements.
                - 'edge_index': Edge indices for the mesh graph.
                - 'edge_attr': Edge attributes for the mesh graph.

        Returns:
            output (Tensor): Final classification output.
            concatenated_features (Tensor): Concatenated aligned features from all modalities.
        """
        # Extract input data
        video_A2C = x['A2C_video']
        video_A4C = x['A4C_video']
        mesh = x['mesh']
        measurements = x['measurements']
        edge_index = x['edge_index']
        edge_attr = x['edge_attr']

        # Initialize an empty tensor to store concatenated features
        concatenated_features = torch.tensor([]).to(video_A2C.device)

        # Process A2C video modality
        if 'A2C' in self.modalities:
            A2C_features = self.video_encoder_A2C(video_A2C)  # Extract features using ResNet3D
            A2C_features_aligned = self.align_A2C(torch.flatten(A2C_features, start_dim=1))  # Align features
            concatenated_features = torch.cat([concatenated_features, A2C_features_aligned], dim=-1)  # Concatenate

        # Process A4C video modality
        if 'A4C' in self.modalities:
            A4C_features = self.video_encoder_A4C(video_A4C)  # Extract features using ResNet3D
            A4C_features_aligned = self.align_A4C(torch.flatten(A4C_features, start_dim=1))  # Align features
            concatenated_features = torch.cat([concatenated_features, A4C_features_aligned], dim=-1)  # Concatenate

        # Process mesh modality
        if 'mesh' in self.modalities:
            mesh_features = self.graph_encoder_mesh(mesh, edge_index, edge_attr)  # Extract features using TGCN
            mesh_features_aligned = self.align_mesh(mesh_features)  # Align features
            concatenated_features = torch.cat([concatenated_features, mesh_features_aligned], dim=-1)  # Concatenate

        # Process clinical measurements modality
        if 'clinical' in self.modalities:
            measurements_aligned = self.measurement_encoder(measurements)  # Align clinical measurements
            concatenated_features = torch.cat([concatenated_features, measurements_aligned], dim=-1)  # Concatenate

        # Perform final classification
        output = self.classifier(concatenated_features)

        return output, concatenated_features
