import torch
import torch.nn as nn


# =========================
# Currin transforms
# =========================


class CurrinDescriptorTransform_1(nn.Module):
    def forward(self, x):
        y = x.clone()
        y[..., 1] = (torch.exp(5.0 * x[..., 1]) - 1.0) / (
            torch.exp(torch.tensor(5.0)) - 1.0
        )
        return y


class CurrinDescriptorTransform_2(nn.Module):
    def forward(self, x):
        y = x.clone()
        y[..., 1] = x[..., 1] ** 0.3
        return y


# =========================
# Branin transforms
# =========================


class BraninDescriptorTransform_1(nn.Module):
    def forward(self, x):
        y = x.clone()

        x1 = 15.0 * x[..., 0] - 5.0
        x2 = 15.0 * x[..., 1]

        valley = x2 - (5.1 / (4.0 * torch.pi**2)) * x1**2 + (5.0 / torch.pi) * x1 - 6.0

        y[..., 1] = valley
        return y


class BraninDescriptorTransform_2(nn.Module):
    def forward(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        t = 2.0 * torch.pi * x1

        y = torch.stack([torch.sin(t), x2], dim=-1)
        return y


class BraninDescriptorTransform_3(nn.Module):
    is_one_to_many = True
    dim=3
    def forward(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        t = 2.0 * torch.pi * x1

        y = torch.stack([torch.sin(t), torch.cos(t), x2], dim=-1)
        return y

