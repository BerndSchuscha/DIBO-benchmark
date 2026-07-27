import torch
from pibob.problems.EnergyDensity.EnergyDensity_Problem import EnergyDensity_Problem
from pibob.models.MyInputTransform import PoolDescriptorTransform

problem = EnergyDensity_Problem()


class EnergyTransform(PoolDescriptorTransform):
    dim = 6
    is_one_to_many = True

    def __init__(self, descriptor_name="M_W_recover_cor_0"):
        super().__init__(
            problem=problem,
            descriptor_name=descriptor_name,
        )

    def forward(self, x):
        y_pool = self.lookup_descriptor(x)
        return torch.cat([x, y_pool[..., None]], dim=-1)
