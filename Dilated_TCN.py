"""
Temporal Convolutional Network (Dilated TCN) Implementation

This module implements Dilated TCN as described in:

    C. Lea, M. D. Flynn, R. Vidal, A. Reiter, G. D. Hager,
    “Temporal Convolutional Networks for Action Segmentation and Detection,”
    in Proceedings of the IEEE Conference on Computer Vision and Pattern
    Recognition (CVPR), 2017.

You can find the paper at:
https://arxiv.org/abs/1611.05267
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from typing import List, Optional, Tuple


class NormalizedReLU(nn.Module):
    def __init__(self, eps: float=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x): # B C T
        y = F.relu(x)
        m = torch.amax(y, dim=1, keepdim=True)
        m = m.clamp(min=self.eps)
        return y / m
        


class DilatedConv1dLayer(nn.Module):
    """
    One dilated causal convolution layer with a residual connection.
    This is the basic layer used inside a TCN block.

    Parameters
    ----------
    in_channels : int
        Dimensionality of the incoming temporal features.
    out_channels : int
        Number of output channels; kept constant across the whole network.
    dilation : int
        Dilation factor s_l = 2^(l-1) for layer index l.
    kernel_size : int, default 2
        Kernel length. Use 3 if you want the acausal variant that
        sees one future frame.
    causal : bool, default True
        If True the padding keeps the output aligned with past
        timesteps only (online or real-time). If False symmetric padding
        forms the acausal setting (offline processing).
    """

    def __init__(
        self,
        f_w: int,
        *,
        dilation: int = 1,
        kernel_size: int = 3,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.left_pad = (kernel_size - 1) * dilation

        pad = 0 if causal else ((kernel_size - 1) * dilation) // 2
        self.filter_conv = nn.Conv1d(
            f_w,
            f_w,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            bias=True,
        )
        self.residual_conv = nn.Conv1d(
            f_w,
            f_w,
            kernel_size=1,
            bias=True
        )
        self.act = NormalizedReLU()
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        """
        Run one dilated convolution step.

        Shape
        -----
        x : (batch, channels, timesteps)
        return : same shape as x (residual addition)
        """
        x_padded = F.pad(x, (self.left_pad, 0)) if self.causal else x
        s = self.filter_conv(x_padded)
        s_hat = self.residual_conv(s)
        y = x + s_hat
        return y, s_hat


class ResidualBlock(nn.Module):
    """
    One TCN block consisting of L dilated layers and an
    internal skip connection.

    Parameters
    ----------
    channels : int
        Fixed hidden width F_w.
    num_layers : int
        L, how many dilated layers to stack inside this block.
    causal : bool, default True
        Toggles causal or acausal mode for every contained layer.
    """

    def __init__(
        self,
        f_w: int,
        num_layers: int,
        *,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DilatedConv1dLayer(
                    f_w,
                    dilation=2 ** l,
                    causal=causal,
                    dropout_p=dropout_p,
                )
                for l in range(num_layers)
            ]
        )
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip_total: Optional[torch.Tensor] = None
        out = x
        for layer in self.layers:
            out, skip = layer(out)
            skip_total = skip if skip_total is None else skip_total + skip
        return out, skip_total  # residual output, aggregated skip


class DilatedTCN(nn.Module):
    """Dilated Temporal Convolutional Network (Lea et al., 2017).

    Notes
    -----
    *Soft‑max is purposely **omitted** so that the module outputs raw **logits**
    suitable for `nn.CrossEntropyLoss` or downstream CTC/CRF heads.*
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        f_w: int = 128,
        num_blocks: int = 4,
        layers_per_block: int = 5,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        # 1) Optional 1×1 projection to hidden width
        self.pre = (
            nn.Identity()
            if input_dim == f_w
            else nn.Conv1d(input_dim, f_w, kernel_size=1, bias=True)
        )

        # 2) Residual blocks stack
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    f_w,
                    num_layers=layers_per_block,
                    causal=causal,
                    dropout_p=dropout_p,
                )
                for _ in range(num_blocks)
            ]
        )

        # 3) Head : ReLU → 1×1 (V) → ReLU → 1×1 (U)
        self.V = nn.Conv1d(f_w, f_w, kernel_size=1, bias=True)
        self.act = nn.ReLU()
        self.U = nn.Conv1d(f_w, num_classes, kernel_size=1, bias=True)

    # -------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, input_dim, T).
        return_features : bool, default False
            If *True*, also return the penultimate hidden map *z*.

        Returns
        -------
        logits : torch.Tensor
            Raw class logits, shape (B, num_classes, T).
        z : Optional[torch.Tensor]
            Hidden feature map (B, f_w, T) if *return_features* else *None*.
        """
        # 1) Input projection
        x = self.pre(x)

        # 2) Residual blocks + global skip accumulation
        skip_total: Optional[torch.Tensor] = None
        for block in self.blocks:
            x, skip = block(x)
            skip_total = skip if skip_total is None else skip_total + skip

        # 3) ReLU → V → ReLU → U
        z = F.relu(skip_total)
        z = self.V(z)
        z = self.act(z)
        logits = self.U(z)

        return (logits, z) if return_features else (logits, None)

if __name__ == "__main__":
    B, C, T = 1, 3, 16000
    dummy_input = torch.randn(B, C, T)
    model = DilatedTCN(
        input_dim=C,
        num_classes=10,
    )
    try:
        output, _ = model(dummy_input)
        print(f"output shape : {output.shape}")
    except Exception as e:
        print(e)
    torchinfo.summary(model, input_size=(B, C, T))