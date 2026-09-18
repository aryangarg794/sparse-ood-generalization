import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from sparse_generalization.utils.util_funcs import get_device


class VHyperNet(nn.Module):

    def __init__(
        self,
        output_dim: int,
        num_modes: int = 3,
        num_heads: int = 1,
        encoder_heads: bool = False,
        hidden_features: list = [128, 128],
        act: nn.Module = nn.ReLU,
        device: str | None = None,
        **kwargs,  
    ):
        device = get_device(device)
        super().__init__()
        self.device = device
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.encoder_heads = encoder_heads
        self.output_dim = output_dim

        total_out = output_dim * num_heads if encoder_heads else output_dim

        dims = [num_modes] + list(hidden_features)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), act()]
        layers.append(nn.Linear(dims[-1], total_out))
        self.hyper = nn.Sequential(*layers)

    def get_modes(self):
        return F.one_hot(torch.arange(self.num_modes, device=self.device), self.num_modes).float()

    def forward(self, x: Tensor = None, num_evals: int = 1):
        if num_evals != self.num_modes:
            raise ValueError(f"VHyperNet generates one weight set per mode, got num_evals={num_evals} \
                              != num_modes={self.num_modes}")

        output = self.hyper(self.get_modes()) 

        if self.encoder_heads:
            batch_size = x.size(0)
            output = output.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, output.size(-1))

        log_prob_z = torch.zeros(output.size(0), device=output.device, dtype=output.dtype)
        return output, log_prob_z
