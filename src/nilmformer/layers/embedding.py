#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Embedding Layers
#
#################################################################################################################

import torch
import torch.nn as nn


class ResUnit(nn.Module):
    def __init__(self, c_in, c_out, k=8, dilation=1, stride=1, bias=True):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=k,
                dilation=dilation,
                stride=stride,
                bias=bias,
                padding="same",
            ),
            nn.GELU(),
            nn.BatchNorm1d(c_out),
        )
        if c_in > 1 and c_in != c_out:
            self.match_residual = True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual = False

    def forward(self, x):
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)

            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))


class DilatedBlock(nn.Module):
    def __init__(
        self, c_in=32, c_out=32, kernel_size=8, dilation_list=[1, 2, 4, 8], bias=True
    ):
        super().__init__()

        layers = []
        for i, dilation in enumerate(dilation_list):
            if i == 0:
                layers.append(
                    ResUnit(c_in, c_out, k=kernel_size, dilation=dilation, bias=bias)
                )
            else:
                layers.append(
                    ResUnit(c_out, c_out, k=kernel_size, dilation=dilation, bias=bias)
                )
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x
