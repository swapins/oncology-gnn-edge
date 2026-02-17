import torch


def load_graph_from_tensors(features: torch.Tensor, adjacency: torch.Tensor):
    """
    Wraps raw tensors into graph structure.
    """

    return {
        "features": features,
        "adjacency": adjacency
    }
