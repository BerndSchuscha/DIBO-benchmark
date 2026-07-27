import torch
from pibob.problems.Bainite.Bainite_Problem import Bainite_Problem
from pibob.models.MyInputTransform import PoolDescriptorTransform

problem = Bainite_Problem()


class AddDescriptorsYs(PoolDescriptorTransform):
    dim = 9
    is_one_to_many = True

    def __init__(self, descriptor_name=["fA", "HAGB"]):
        super().__init__(
            problem=problem,
            descriptor_name=descriptor_name,
        )

    def forward(self, x):
        y_pool = self.lookup_descriptor(x)
        return torch.cat([x, y_pool], dim=-1)


class AddDescriptorsUEL(PoolDescriptorTransform):
    dim = 11
    is_one_to_many = True

    def __init__(self, descriptor_name=["fA", "HAGB", "Ys", "Ms_RA"]):
        super().__init__(
            problem=problem,
            descriptor_name=descriptor_name,
        )

    def forward(self, x):
        y_pool = self.lookup_descriptor(x)
        return torch.cat([x, y_pool], dim=-1)


class AddDescriptorsYs1(PoolDescriptorTransform):
    dim = 8
    is_one_to_many = True

    def __init__(self, descriptor_name=["fA"]):
        super().__init__(
            problem=problem,
            descriptor_name=descriptor_name,
        )

    def forward(self, x):
        y_pool = self.lookup_descriptor(x)
        return torch.cat([x, y_pool[..., None]], dim=-1)


class AddDescriptorsUEL1(PoolDescriptorTransform):
    dim = 8
    is_one_to_many = True

    def __init__(self, descriptor_name=["fA"]):
        super().__init__(
            problem=problem,
            descriptor_name=descriptor_name,
        )

    def forward(self, x):
        y_pool = self.lookup_descriptor(x)
        return torch.cat([x, y_pool[..., None]], dim=-1)
