from torch_geometric_temporal.nn.recurrent import TGCN2
from torch import nn
from torchvision.models.video import r3d_18
from config.config import CONFIG


class TemporalMeshGCN(nn.Module):
    """
    Temporal Graph Convolutional Network for processing temporal graph data.

    Args:
        node_features (int): Number of input features per node.
        out_channels (int): Number of output channels.

    Attributes:
        tgnn (TGCN2): Temporal Graph Convolutional Network layer.
        fc (nn.Sequential): Fully connected layers for final output.
    """
    def __init__(self, node_features, out_channels):
        super(TemporalMeshGCN, self).__init__()
        # Temporal Graph Convolutional Network layer
        self.tgnn = TGCN2(node_features, 16, batch_size=4)  # batch_size is required but not used in source code

        # Fully connected layers for final output
        self.fc = nn.Sequential(
            nn.Linear(CONFIG.data.mesh_nodes * 16, 512), 
            nn.LeakyReLU(),
            nn.Linear(512, out_channels),
        )

    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, num_nodes, features).
            edge_index (torch.Tensor): Edge indices shared across all time steps.
            edge_weight (torch.Tensor): Edge weights shared across all time steps.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels).
        """
        time_steps = x.shape[1]
        hidden_state = None  # Initialize the hidden state

        for t in range(time_steps):
            # Process each time step
            x_t = x[:, t, :, :]  # Shape (batch_size, num_nodes, features)
            hidden_state = self.tgnn(x_t, edge_index[0], edge_weight[0], hidden_state)

        # After processing all time steps, apply the linear layer
        h = hidden_state.flatten(1)  # Flatten the hidden state
        h = self.fc(h)  # Pass through fully connected layers
        return h


class SimpleFCN(nn.Module):
    """
    Simple Fully Connected Network for classification or alignment task.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output.

    Attributes:
        fc (nn.Sequential): Fully connected layers.
    """
    def __init__(self, input_dim, output_dim):
        super(SimpleFCN, self).__init__()
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.fc(x)


class VideoEncoder(nn.Module):
    """
    Video Encoder using a pre-trained 3D ResNet model.

    Args:
        resnet_out_dim (int): Dimension of the output features from ResNet.

    Attributes:
        resnet (nn.Module): Pre-trained 3D ResNet model.
        fc (nn.Sequential): Fully connected layer for feature transformation.
    """
    def __init__(self, resnet_out_dim):
        super(VideoEncoder, self).__init__()
        # Pre-trained 3D ResNet model
        self.resnet = r3d_18(weights='DEFAULT')
        # Fully connected layer for feature transformation
        self.fc = nn.Sequential(
            nn.Linear(resnet_out_dim, 128),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input video tensor of shape (batch_size, channels, time, height, width).

        Returns:
            torch.Tensor: Encoded video features of shape (batch_size, 128).
        """
        features = self.resnet(x)  # Extract features using ResNet
        features = self.fc(features)  # Transform features
        return features
