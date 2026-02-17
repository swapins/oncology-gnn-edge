import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseGraphLayer(nn.Module, ABC):
    """
    Abstract base class for graph convolution layers.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: Node feature matrix (N x F)
        adj: Adjacency matrix (N x N)
        """
        raise NotImplementedError("Subclasses must implement `forward(x, adj)`")
