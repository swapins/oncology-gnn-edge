import torch
from gnn_edge.models.normalized_gcn import NormalizedGCNLayer


def test_no_nan_in_normalization():
    layer = NormalizedGCNLayer(32, 16)

    adj = torch.zeros((10, 10))
    x = torch.randn(10, 32)

    output = layer(x, adj)

    assert not torch.isnan(output).any()
