import math
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
        input_dim: int | None = None,
        bias_init: bool = False,
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

        if bias_init:
            # Bias-HyperInit (Beck et al., 2022, arXiv:2210.11348): head weights W := 0 and bias
            # b := phi_shared ~ f(phi), so every mode starts from the same base-network weights. f is the
            # nn.Linear default for the generated (d, d) matrices, U(-1/sqrt(d), 1/sqrt(d)) with d = input_dim
            if input_dim is None:
                raise ValueError("bias_init needs input_dim, the fan-in of the generated matrices")
            head = self.hyper[-1]
            nn.init.zeros_(head.weight)
            nn.init.uniform_(head.bias, -1 / math.sqrt(input_dim), 1 / math.sqrt(input_dim))

    def get_modes(self):
        return F.one_hot(torch.arange(self.num_modes, device=self.device), self.num_modes).float()

    def forward(self, x: Tensor = None):
        output = self.hyper(self.get_modes()) 

        if self.encoder_heads:
            batch_size = x.size(0)
            output = output.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, output.size(-1))

        log_prob_z = torch.zeros(output.size(0), device=output.device, dtype=output.dtype)
        return output, log_prob_z
