import time
from gnn_edge.data.synthetic_ppi import generate_synthetic_ppi
from gnn_edge.config import GNNConfig
from gnn_edge.inference import GNNInference


def benchmark(sparse_mode: bool):

    config = GNNConfig(input_dim=64, hidden_dim=32)
    engine = GNNInference(config)

    graph = generate_synthetic_ppi(500, 64, sparse=sparse_mode)

    start = time.time()
    for _ in range(50):
        engine.forward(graph)
    end = time.time()

    mode = "Sparse" if sparse_mode else "Dense"
    print(f"{mode} mode: {end - start:.4f} seconds")


if __name__ == "__main__":
    benchmark(sparse_mode=False)
    benchmark(sparse_mode=True)
