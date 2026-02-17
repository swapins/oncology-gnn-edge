import time

# Import core gnn_edge modules first (torch-dependent) before PyQt5
from gnn_edge.config import GNNConfig
from gnn_edge.inference import GNNInference
from gnn_edge.data.synthetic_ppi import generate_synthetic_ppi
from gnn_edge.logger import setup_logger, log_system_info

# Import PyQt5 after torch is loaded
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from gnn_edge.ui.dashboard import GNNSystemDashboard


def run_application():

    logger = setup_logger()
    log_system_info(logger)

    config = GNNConfig(
        input_dim=64,
        hidden_dim=32,
        use_fp16=True
    )

    engine = GNNInference(config)

    logger.info(f"Using device: {engine.device}")
    logger.info(f"FP16 enabled: {engine.use_fp16}")

    app = QApplication([])
    dashboard = GNNSystemDashboard(engine)
    dashboard.show()

    def run_inference():

        graph = generate_synthetic_ppi(
            200,
            64,
            sparse=True,
            device=engine.device
        )

        start = time.time()
        output = engine.forward(graph)
        end = time.time()

        inference_time = (end - start) * 1000

        logger.info(
            f"Inference completed | "
            f"Nodes={graph['adjacency'].size(0)} | "
            f"Time={inference_time:.2f} ms"
        )

        dashboard.update_inference_info(
            graph,
            output,
            inference_time
        )

    timer = QTimer()
    timer.timeout.connect(run_inference)
    timer.start(2000)

    app.exec_()


if __name__ == "__main__":
    run_application()
