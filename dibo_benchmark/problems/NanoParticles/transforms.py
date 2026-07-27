import torch
import torch.nn as nn
class NanoDescriptorTransform_2(nn.Module):
    """
    Compressed shared transform for both nanoparticle objectives.
    """

    is_one_to_many = True
    dim = 8

    def forward(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        x3 = x[..., 2]
        x4 = x[..., 3]

        y = torch.stack(
            [
                x1,
                x2,
                x3,
                x4,
                x1 * x2,
                x1 * x3,
                x1 * x4,
                x4**2,
            ],
            dim=-1,
        )
        return y