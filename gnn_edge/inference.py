import torch
from gnn_edge.models.normalized_gcn import NormalizedGCNLayer


class GNNInference:

    def __init__(self, config):

        self.config = config
        self.device = config.resolve_device()

        self.layer = NormalizedGCNLayer(
            config.input_dim,
            config.hidden_dim
        ).to(self.device)

        self.layer.eval()

        # Only enable FP16 if running on CUDA
        self.use_fp16 = config.use_fp16 and self.device.type == "cuda"

        if self.use_fp16:
            self.layer = self.layer.half()

    @torch.no_grad()
    def forward(self, graph_data):

        x = graph_data["features"].to(self.device)
        adj = graph_data["adjacency"].to(self.device)

        if self.use_fp16:
            x = x.half()
            if not adj.is_sparse:
                adj = adj.half()

        return self.layer(x, adj)
