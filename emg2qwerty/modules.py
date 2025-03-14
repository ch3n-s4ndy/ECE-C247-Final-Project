# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F



class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: Sequence[int], dropout: float = 0.1, use_residual: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        
        lstm_blocks = []

        lstm_blocks.append(
            nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=layers[0],
                dropout=dropout if layers[0] > 1 else 0,
                bidirectional=True
            )
        )

        for num in layers[1:]:
            lstm_blocks.append(
                nn.LSTM(
                    input_size=hidden_dim * 2,
                    hidden_size=hidden_dim,
                    num_layers=num,
                    dropout=dropout if num > 1 else 0,
                    bidirectional=True
                )
            )
        
        self.lstm_layers = nn.ModuleList(lstm_blocks)
        self.projection = nn.Linear(hidden_dim * 2, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm_layers[0](x)
        for lstm in self.lstm_layers[1:]:
            new_out, _ = lstm(out)
            if self.use_residual:
                out = out + new_out
            else:
                out = new_out
        final = self.projection(out)
        return final


########################################################
class LSTMConvBlock(nn.Module):
    def __init__(self, input_size: int, lstm_hidden_size: int, lstm_num_layers: int, conv_kernel: int, dropout: float = 0.1, residual_connection: bool = True) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers=lstm_num_layers, dropout=dropout, bidirectional=True)
        self.projection = nn.Linear(lstm_hidden_size * 2, input_size)
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=conv_kernel, padding=conv_kernel // 2)
        self.relu = nn.ReLU()
        self.residual_connection = residual_connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)                
        lstm_out = self.projection(lstm_out)        
        conv_in = lstm_out.permute(1, 2, 0)         
        conv_out = self.conv(conv_in)              
        conv_out = self.relu(conv_out)
        conv_out = conv_out.permute(2, 0, 1)          
        if self.residual_connection:
            return x + conv_out
        else:
            return conv_out


########################################################
class MultiLayerLSTMConvEncoder(nn.Module):
    def __init__(self, input_size: int, block_params: Sequence[dict]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            LSTMConvBlock(
                input_size=input_size,
                lstm_hidden_size=bp.get("lstm_hidden_size", 128),
                lstm_num_layers=bp.get("lstm_num_layers", 1),
                conv_kernel=bp.get("conv_kernel", 3),
                dropout=bp.get("dropout", 0.1),
                residual_connection=bp.get("residual_connection", True),
            ) for bp in block_params
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


########################################################
class TransformerEncoder(nn.Module):
    def __init__(self, in_features, num_heads, num_layers, d_model, ff_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(in_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, in_features)
    
    def forward(self, x):
        x = self.embedding(x)      
        x = self.transformer(x)     
        x = self.fc(x)             
        return x


########################################################
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_kernel=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


########################################################
class CNNTransformerBlock(nn.Module):
    def __init__(self, channels, transformer_params, cnn_params, residual_connection: bool = True):
        super(CNNTransformerBlock, self).__init__()
        self.residual_connection = residual_connection
        self.cnn = CNNBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=cnn_params.get("kernel_size", 3),
            padding=cnn_params.get("padding", 1),
            pool_kernel=cnn_params.get("pool_kernel", 2)
        )
        self.transformer = TransformerEncoder(
            in_features=channels,
            num_heads=transformer_params.num_heads,
            num_layers=transformer_params.num_layers,
            d_model=transformer_params.d_model,
            ff_dim=transformer_params.ff_dim,
            dropout=transformer_params.dropout,
        )
    
    def forward(self, x):
        residual = x  
        x = self.cnn(x)  
        T_out = x.shape[-1]
        residual = residual[..., -T_out:]
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)     
        if self.residual_connection:
            x = x + residual
        return x


########################################################
class StackedCNNTransformer(nn.Module):
    def __init__(self, num_blocks, channels, transformer_params, cnn_params, residual_connection: bool = True):
        super(StackedCNNTransformer, self).__init__()
        self.blocks = nn.ModuleList([
            CNNTransformerBlock(channels, transformer_params, cnn_params, residual_connection=residual_connection)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        # x: (batch, channels, T)
        for block in self.blocks:
            x = block(x)
        return x


########################################################
class RearrangeMLPOutput(nn.Module):
    def forward(self, x):
        """
        input：(T, N, bands, mlp_features)
        output：(N, bands * mlp_features, T)
        """
        x = x.permute(1, 2, 3, 0)
        batch, bands, feat, T = x.shape
        return x.reshape(batch, bands * feat, T)


########################################################
class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)