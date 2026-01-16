import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)

def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.layers(x)

class ResNetBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=1
            ) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1
                ),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)

class ResNetRepresentation(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int = 4) -> None:
        super().__init__()

        self.input_args = {
            'in_channels': in_channels,
        }
            
        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),
        ])

    def forward(self, x: torch.Tensor):
        
        z = self.layers(x)
        z = z.mean(dim=-1)
        return z

def resnet_ts(**kwargs):
    return {'backbone': ResNetRepresentation(**kwargs), 'dim': kwargs['mid_channels']*2}


class TransformerRepresentation(nn.Module):
    def __init__(self, in_channels, d_model=8, nhead=2, num_layers=2, dim_feedforward=16, dropout=0.1, recon_hidden=32):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model
        self.in_channels = in_channels
        self.recon_hidden = recon_hidden
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, recon_hidden),
            nn.ReLU(),
            nn.Linear(recon_hidden, in_channels)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        out = x.mean(dim=1)
        recon = self.reconstruction_head(out)
        return out, recon

def transformer_ts(**kwargs):
    transformer_kwargs = {
        'in_channels': kwargs['in_channels'],
        'd_model': kwargs.get('d_model', 8),
        'nhead': kwargs.get('nhead', 2),
        'num_layers': kwargs.get('num_layers', 2),
        'dim_feedforward': kwargs.get('dim_feedforward', 16),
        'dropout': kwargs.get('dropout', 0.1),
        'recon_hidden': kwargs.get('recon_hidden', 32)
    }
    return {
        'backbone': TransformerRepresentation(**transformer_kwargs),
        'dim': transformer_kwargs['d_model']
    }