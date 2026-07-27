import torch
from pibob.models.MyInputTransform import PoolDescriptorTransform
from pibob.problems.PS_ALSC.PS_ALSC_Problem import PS_ALSC_Problem
import torch.nn as nn

problem = PS_ALSC_Problem()


class ALSCTransform(nn.Module):
    dim = 5

    def forward(self, x):
        x1 = 1 / (x[..., 0] + 3)
        x2 = 1 / (x[..., 1] + 3)
        x3 = torch.sqrt(x[..., 2] + 1e-16)
        x4 = torch.sqrt(x[..., 3] + 1e-16)
        x5 = x[..., 4]

        y = torch.stack([x1, x2, x3, x4, x5], dim=-1)

        if torch.isnan(x3).any():
            print(x[..., 2])

        return y


class ALSCTransform1(PoolDescriptorTransform):
    dim = 6
    is_one_to_many = True

    def __init__(self, descriptor_name=""):
        super().__init__(
            problem=problem,
            descriptor_name=descriptor_name,
        )

    def forward(self, x):
        y_pool = self.lookup_descriptor(x)
        return torch.cat([x, y_pool[..., None]], dim=-1)
