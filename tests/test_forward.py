import torch
from gnn_edge.config import GNNConfig
from gnn_edge.inference import GNNInference
from gnn_edge.data.synthetic_ppi import generate_synthetic_ppi


def test_forward_pass_runs():
    config = GNNConfig(input_dim=64, hidden_dim=32)
    engine = GNNInference(config)

    graph = generate_synthetic_ppi(50, 64)
    output = engine.forward(graph)

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 50
