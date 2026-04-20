import lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
from typing import List, Self


class BasicCNN(nn.Module):
    """Basic CNN class, based

    Args:
        input_dim (int): Size of Input.
        out_dim (int): Size of output.
        hidden_dims (List): Model architecture.
        act (nn.Module): Activation function.
    """

    def __init__(
        self: Self,
        input_dim: int,
        out_dim: int,
        embedding_inp: bool,
        hidden_dims: List,
        act: nn.Module,
        dropout: float,
        num_embeddings: int = 64,
        model_dim: int = 32,
        inp_size: int = 10,
        *args,
        **kwargs,
    ):
        super(BasicCNN, self).__init__(*args, **kwargs)
        self.embedding_inp = embedding_inp
        if embedding_inp:
            self.feature_map = nn.Embedding(num_embeddings, model_dim)

        self.layers = nn.Sequential()
        self.layers.extend(
            [
                nn.Conv2d(
                    in_channels=model_dim if embedding_inp else input_dim,
                    out_channels=hidden_dims[0],
                    kernel_size=3,
                    padding=1,
                ),
                act(),
            ]
        )

        for ch1, ch2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.extend(
                [
                    nn.Conv2d(
                        in_channels=ch1, out_channels=ch2, kernel_size=3, padding=1
                    ),
                    act(),
                ]
            )

        self.layers.extend(
            [
                nn.MaxPool2d(kernel_size=2, stride=2),
                act(),
            ]
        )

        with torch.no_grad():
            size = (1, model_dim if embedding_inp else input_dim, inp_size, inp_size)
            test_inp = torch.randn(size=(size))
            out = self.layers(test_inp)

        self.ffn = nn.Sequential()
        self.ffn.extend(
            [
                nn.Linear(
                    in_features=out.size(2) * out.size(3) * hidden_dims[-1],
                    out_features=64,
                ),
                act(),
                nn.Dropout(dropout),
                nn.Linear(in_features=64, out_features=64),
                act(),
                nn.Linear(in_features=64, out_features=out_dim),
            ]
        )

    def forward(self: Self, x: Tensor):
        if self.embedding_inp:
            x = self.feature_map(x.squeeze(3).int())
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.flatten(start_dim=1)
        x = self.ffn(x)
        return x
