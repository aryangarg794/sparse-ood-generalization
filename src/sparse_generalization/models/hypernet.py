import torch 
import torch.nn as nn
import zuko

class HyperNet(nn.Module):

    def __init__(
            self, 
            num_layers: int = 1, 
            seq_len: int = 1, 
            latent_dim: int = 16, 
            num_heads: int = 1, 
            lstm_layers: int = 1, 
            dropout: float = 0.0, 
            bidirectional: bool = True, 
            flow_params: dict = {'n_flows' : 2, 'hidden_features' : (128, 128)},
            prior_params: dict = {'n_flows' : 3, 'hidden_features' : (256, 256)},
            residual: bool = False,
            nf_prior: bool = True, 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)

class HyperNetSpartan(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)