from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SincNetConfig:
    """The default values are adapted from Bravo Sanchez et al. (2021) tuned for NIPS4Bplus"""

    # Data
    batch_size: int = 128

    # Windowing parameters
    sample_rate: int = 44100
    cw_len: int = 18 # window length in ms
    cw_shift: int = 1 # overlap in ms

    # number of filters, kernel size, stride
    conv_layers: List[Tuple[int, int, int]] = (
        (220, 151, 1),
        (60, 5, 1),
        (60, 5, 1),
    )
    # set any of these < 1 to disable max pooling for the conv layer
    conv_max_pool_len: Tuple[int] = (5, 5, 5)

    # if batchnorm is not used, layernorm is applied instead
    conv_layers_batchnorm: bool = True

    # number of neurons
    fc_layers: List[int] = (
        1024,
        1024,
        1024,
    )

    fc_layers_batchnorm: bool = True

    # number of classes
    num_classes: int = 87

class SincNetModel(nn.Module):
    """A model of 3 components: Convolution, MLP, Linear classification"""
    def __init__(self, cfg: SincNetConfig):
        super().__init__()

        self.cfg = cfg

        self.conv_layers = nn.ModuleList([])
        self.fc_layers = nn.ModuleList([])
        self.classification_layer = nn.Linear(in_features=cfg.fc_layers[-1], out_features=cfg.num_classes)

        # Calculate the output size of the last conv layer
        self.conv_output_size = self.calculate_conv_output_size()

        # create the conv layers
        for i, (out_channels, kernel_size, stride) in enumerate(cfg.conv_layers):
            if i == 0:
                self.conv_layers.append(nn.Sequential(
                    SincConv(out_channels=out_channels, sample_rate=cfg.sample_rate, kernel_size=kernel_size),
                    # todo: if using layernorm instead of batchnom, maxpool should be applied to the torch.abs of the sinc layer.
                    nn.MaxPool1d(cfg.conv_max_pool_len[i]) if cfg.conv_max_pool_len[i] > 1 else nn.Identity(),
                    ChannelwiseLayerNorm(out_channels) if not cfg.conv_layers_batchnorm else nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU()
                ))
            else:
                in_channels = cfg.conv_layers[i-1][0]
                self.conv_layers.append(nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                    nn.MaxPool1d(cfg.conv_max_pool_len[i]) if cfg.conv_max_pool_len[i] > 1 else nn.Identity(),
                    ChannelwiseLayerNorm(out_channels) if not cfg.conv_layers_batchnorm else nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU()
                ))

        # create the fc layers
        for i, out_features in enumerate(cfg.fc_layers):
            in_features = self.conv_output_size if i == 0 else cfg.fc_layers[i-1]
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LayerNorm(out_features) if not cfg.fc_layers_batchnorm else nn.BatchNorm1d(out_features),
                # nn.Dropout(0.5), # experimental
                nn.LeakyReLU()
            ))


    def calculate_conv_output_size(self):
        """Helper function to calculate the output size of the last conv layer, accounting for padding and max pooling"""
        # Start with the initial input size in samples
        size = self.cfg.cw_len * self.cfg.sample_rate // 1000
        channels = 1

        # Iterate through each conv layer to calculate the output size
        for i, (out_channels, kernel_size, stride) in enumerate(self.cfg.conv_layers):
            if i == 0:
                # SincConv layer uses "same" padding
                padding = (kernel_size - 1) // 2
            else:
                # Assuming no padding for standard Conv1d layers; modify if padding is added
                padding = 0

            # Calculate size after convolution
            size = (size + 2 * padding - kernel_size) // stride + 1

            # Apply max pooling if specified
            if self.cfg.conv_max_pool_len[i] > 1:
                size = size // self.cfg.conv_max_pool_len[i]

            channels = out_channels

        return channels * size

    def forward(self, x):
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            # x = conv_layer[0](x)  # Apply Conv1d or SincConv
            # x = x.transpose(1, 2)  # Transpose: (batch, channels, time) -> (batch, time, channels)
            # x = conv_layer[1](x)  # Apply LayerNorm
            # x = x.transpose(1, 2)  # Transpose back: (batch, time, channels) -> (batch, channels, time)
            # x = conv_layer[2](x)  # Apply LeakyReLU

        # Flatten the output for fully connected layers
        x = x.contiguous().view(x.size(0), -1)

        # Apply fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # Apply classification layer
        logits = self.classification_layer(x)

        return logits
#----------------------------------------------------------------------------
class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x shape: (batch, channels, time)
        # Transpose, normalize, and transpose back
        return self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
#----------------------------------------------------------------------------
class SincConv(nn.Module):
    def __init__(self, out_channels: int, sample_rate: int, kernel_size: int, min_low_hz: int = 50, min_band_hz: int = 50):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Typically constant values
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Convert these to tensors
        self.low_hz_ = torch.tensor(30.0)
        self.high_hz_ = torch.tensor(self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz))

        # Initialize mel-spaced filterbanks
        mel = torch.linspace(self.to_mel(self.low_hz_), self.to_mel(self.high_hz_), self.out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(hz[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).unsqueeze(1))

        # Compute Hamming window (constant)
        window_ = torch.hamming_window(self.kernel_size, periodic=False)
        self.register_buffer('window_', window_)
        # Compute time axis (constant)
        n_ = (2 * torch.pi * torch.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1)).unsqueeze(0) / self.sample_rate
        self.register_buffer('n_', n_)

    @staticmethod
    def to_mel(hz):
        return 2595 * torch.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, waveforms):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high - low)[:, 0]

        f_times_t = torch.outer(low.squeeze(), self.n_.squeeze())
        
        # Compute band-pass filter
        sin_term1 = torch.sin(f_times_t + band.unsqueeze(1) * self.n_)
        sin_term2 = torch.sin(f_times_t)
        
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-8
        band_pass = (sin_term1 - sin_term2) / (self.n_ + epsilon)        
        band_pass *= self.window_.to(waveforms.device)

        # Normalize and reshape filters

        # -- Add epsilon to avoid division by zero --
        band_pass = band_pass / (2 * band.unsqueeze(1) + epsilon)
        
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, filters, stride=1, padding=(self.kernel_size - 1) // 2)
