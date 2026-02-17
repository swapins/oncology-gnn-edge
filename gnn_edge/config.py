from dataclasses import dataclass
import torch


@dataclass
class GNNConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int = 16
    use_normalized: bool = True
    force_device: str | None = None   # "cpu" or "cuda" or None
    use_fp16: bool = False

    def resolve_device(self) -> torch.device:
        """
        Automatically determine device unless explicitly forced.
        """

        if self.force_device:
            return torch.device(self.force_device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")
