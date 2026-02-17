import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class OncologyGNN(nn.Module):
    """
    Normalized GNN for Protein-Protein Interaction (PPI) Analysis.
    Reference: Vidya, S. (2026). Edge-Based Execution of GNNs in Oncology.
    """
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super(OncologyGNN, self).__init__()
        # Symmetric Degree Normalization: H = σ(D^-1/2 A D^-1/2 H W)
        # This formulation bounds the spectral radius for edge stability.
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        
        # Multi-Layer Perceptron (MLP) for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index, batch):
        # 1. Normalized Graph Convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 2. Graph-Level Readout (Global Mean Pooling)
        # z = 1/|V| * sum(h_v)
        z = global_mean_pool(x, batch)

        # 3. MLP Head for final prediction (e.g., Treatment Response)
        return self.classifier(z)