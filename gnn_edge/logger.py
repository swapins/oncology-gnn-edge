import logging
from logging.handlers import RotatingFileHandler
import os
import torch
import platform


def setup_logger(log_dir: str = "logs",
                 level: int = logging.INFO) -> logging.Logger:

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("GNN_EDGE")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "gnn_edge.log"),
        maxBytes=5_000_000,  # 5MB
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_system_info(logger: logging.Logger):

    logger.info("===== System Information =====")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"Total GPU Memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
        )

    logger.info("================================")
