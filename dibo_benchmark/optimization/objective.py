from typing import Optional, Tuple
import torch


class Objective:
    """Objective backed by a precomputed lookup table with reproducible noise."""

    def __init__(
        self,
        Y_table: torch.Tensor,
        noise_seed: int = 0,
    ):
        if Y_table.ndim != 2:
            raise ValueError("Y_table must have shape (N, m).")

        self.Y_table = Y_table
        self.noise_seed = int(noise_seed)

        # --- precompute standard normal noise for every point ---
        gen = torch.Generator(device=Y_table.device)
        gen.manual_seed(self.noise_seed)

        self.noise_table = torch.randn(
            *Y_table.shape,
            generator=gen,
            device=Y_table.device,
            dtype=Y_table.dtype,
        )

    def evaluate(
        self,
        idx: torch.Tensor,
        noise_se: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        Y_true = self.Y_table[idx]

        if noise_se is None:
            return Y_true, Y_true

        noise = self.noise_table[idx]
        Y_noisy = Y_true + noise * noise_se
        return Y_noisy, Y_true
