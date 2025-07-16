"""
Conditional PixelCNN Implementation

This module implements Conditional PixelCNN, extended with **spatially-aware
conditioning** (Sec. 3-2, “location-dependent conditional information”) as
described in:

    A. van den Oord, N. Kalchbrenner, O. Vinyals, L. Espeholt,
    A. Graves, K. Kavukcuoglu,
    “Conditional Image Generation with PixelCNN Decoders,”
    Advances in Neural Information Processing Systems 29 (NeurIPS 2016).

Paper available at:
https://arxiv.org/abs/1606.05328
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchinfo
import math


class MaskedConv2d(nn.Conv2d):
    """
    2-D convolution with an autoregressive mask (Type-A or Type-B).

    Parameters
    ----------
    mask_type : str
        `"A"` for the first layer (excludes current pixel) or
        `"B"` for subsequent layers (includes current pixel).
    in_channels, out_channels, kernel_size, dilation, **kwargs
        Standard ``nn.Conv2d`` arguments.

    Notes
    -----
    *The mask is registered as a non-trainable buffer so it is moved across
    devices with the module.*
    """

    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        dilation: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            **kwargs,
        )
        # internal helpers
        self._mask_type = mask_type
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self._build_mask()

    def _build_mask(self) -> None:
        """Create and register the binary autoregressive mask."""
        self.mask.data.fill_(1)
        _, _, kH, kW = self.weight.shape
        cy, cx = kH // 2, kW // 2
        self.mask[:, :, cy + 1:, :] = 0
        self.mask[:, :, cy, cx + 1:] = 0
        if self._mask_type == "A":
            self.mask[:, :, cy, cx] = 0
        



    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Apply masked convolution."""
        masked_weight = self.weight * self.mask
        return F.conv2d(
            x,
            masked_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class GatedActivationUnit(nn.Module):
    """
    Gated activation unit: ``tanh(A) * sigmoid(B)``.

    Notes
    -----
    This is Eq. (2) in the paper and is applied identically to the
    vertical and horizontal stacks.
    """

    def __init__(self, split_dim=1) -> None:
        super().__init__()
        self.split_dim = split_dim

    def forward(self, a: Tensor, b: Tensor | None = None) -> Tensor:  # noqa: D401
        """Combine two feature tensors with gating."""
        if b is None:
            a, b = a.chunk(2, dim=self.split_dim)
        return torch.tanh(a) * torch.sigmoid(b)


class GatedPixelCNNBlock(nn.Module):
    """
    One dual-stack masked convolutional block with gating & residual link.

    Parameters
    ----------
    channels : int
        Feature map width ``p`` in the paper.
    kernel_size : int, default 5
        Square kernel size for the vertical stack. Horizontal stack uses
        ``(1, kernel_size)``.
    dilation : int, default 1
        Dilation factor for both stacks.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 5,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.vertical = MaskedConv2d(
            mask_type="B",
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size // 2) * dilation,
        )
        self.horizontal = MaskedConv2d(
            mask_type="B",
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding=(0, (kernel_size // 2) * dilation), # 상하 0 좌우만 패딩
        )
        self.vert_to_horiz = nn.Conv2d(
            2 * channels,
            2 * channels,
            kernel_size=1,
            bias=True,
        )
        self.gate = GatedActivationUnit()
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
            self,
            v: Tensor,
            h: Tensor,
            cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Returns
        -------
        tuple
            Updated vertical and horizontal stack feature maps.
        """
        # masked conv
        ver = self.vertical(v) # B, 2C, H, W
        hor = self.horizontal(h) # B, 2C, H, W

        # cond bias injection
        if cond is not None:
            ver = ver + cond
            hor = hor + cond
        
        # vertical to horizontal
        hor = hor + self.vert_to_horiz(ver)
        
        # gating
        v_out = self.gate(ver)
        h_out = self.gate(hor)

        # residual
        h_out = self.residual(h_out) + h
        return v_out, h_out


class ConditioningNetwork(nn.Module):
    """
    Upsamples a global condition vector *h* into a spatial feature map *s*.

    This module realises Eq. (4) in Sec. 3-2.
    """

    def __init__(
        self,
        cond_dim: int,
        out_channels: int,
        target_resolution: int,
        base_channels: int = 256,
    ) -> None:
        super().__init__()
        
        # 2의 거듭제곱임을 확인하는 비트 연산
        if target_resolution < 4 or target_resolution & (target_resolution - 1):
            raise ValueError("target_resolution must be power of 2")

        # 선형 투사
        self.init_resolution = 4
        self.init_channels = base_channels
        self.fc = nn.Linear(cond_dim, self.init_channels * self.init_resolution**2)

        # 업샘플
        n_upsamples = int(math.log2(target_resolution // self.init_resolution))
        modules: list[nn.Module] = []
        in_c = self.init_channels
        for _ in range(n_upsamples):
            out_c = max(in_c // 2, out_channels)
            modules.extend(
                [
                    nn.ConvTranspose2d(
                        in_c, out_c,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
            )
            in_c = out_c
        
        # 채널 맞추기
        modules.append(
            nn.Conv2d(
                in_c, out_channels,
                kernel_size=1
            )
        )
        self.upsampler = nn.Sequential(*modules)

    def forward(self, h: Tensor) -> Tensor:  # noqa: D401
        """Project condition vector to spatial map ``s``."""
        B = h.size(0)

        x = self.fc(h).view(
            B,
            self.init_channels,
            self.init_resolution,
            self.init_resolution,
        )

        s = self.upsampler(x)
        return s


class ConditionalPixelCNN(nn.Module):
    """
    Complete Conditional PixelCNN through spatial conditioning.

    Parameters
    ----------
    image_channels : int
        Usually 3 (RGB).
    channels : int, default 128
        Feature width *p* for all gated blocks.
    num_blocks : int, default 15
        Number of residual gated blocks.
    condition_dim : int | None, default None
        Dimensionality of the global condition vector *h*.
    spatial_condition : bool, default False
        If True, use a learnable upsampling network to inject location-dependent
        bias as in Sec. 3-2.
    """

    def __init__(
        self,
        image_channels: int,
        *,
        channels: int = 128,
        num_blocks: int = 15,
        condition_dim: int | None = None,
        spatial_condition: bool = False,
        img_resolution: int = 32,
    ) -> None:
        super().__init__()
        # Mask-A first layer (separate for RGB channel ordering)
        self.input_conv = MaskedConv2d(
            mask_type="A",
            in_channels=image_channels,
            out_channels=channels,
            kernel_size=7,
            padding=3,
        )

        # Gated residual stack
        self.blocks = nn.ModuleList(
            [
                GatedPixelCNNBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # 1×1 conv to logits
        self.out_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels, image_channels * 256, kernel_size=1),
        )

        # Conditioning paths
        self.has_cond = condition_dim is not None
        self.use_spatial = spatial_condition
        if self.has_cond and self.use_spatial:
            self.cond_proj = ConditioningNetwork(
                condition_dim, channels * 2, img_resolution
            )
        elif self.has_cond:
            self.cond_proj = nn.Linear(condition_dim, channels * 2)

    def _inject_condition(
        self,
        h_vec: Tensor | None,
        *,
        batch: int,
        height: int,
        width: int,
    ) -> Tensor:
        """
        Transform and broadcast condition vector ``h`` for addition into
        each gated block.
        """
        if h_vec is None:
            return None
        if self.use_spatial:
            cond = self.cond_proj(h_vec)
        else:
            cond = self.cond_proj(h_vec)
            cond = cond.view(batch, -1, 1, 1).expand(-1, -1, height, width)
        return cond

    def forward(
        self,
        x: Tensor,
        h_vec: Tensor | None = None,
    ) -> Tensor:  # noqa: D401
        """
        Parameters
        ----------
        x : Tensor
            Input image tensor ``[B, C, H, W]`` scaled to [0, 1].
        h : Tensor | None
            Optional condition vector ``[B, d]``.

        Returns
        -------
        Tensor
            Logits of shape ``[B, C, 256, H, W]``.
        """
        B, _, H, W = x.shape

        v = self.input_conv(x)
        h = v.clone()

        cond = self._inject_condition(h_vec, batch=B, height=H, width=W)

        for block in self.blocks:
            v, h = block(v, h, cond=cond)
        
        out = self.out_head(h)
        C_img = self.input_conv.in_channels

        logits = out.view(B, C_img, 256, H, W)
        return logits


if __name__=="__main__":
    B, C, H, W = 1, 3, 32, 32
    cond_dim = 128

    dummy_x = torch.randn(B, C, H, W)
    dummy_h_vec = torch.randn(B, cond_dim)

    model = ConditionalPixelCNN(
        3,
        condition_dim=cond_dim,
        spatial_condition=True,
        img_resolution=H,
    )
    logits = model(dummy_x, dummy_h_vec)
    print(f"output shape: {logits.shape}")
    
    torchinfo.summary(model, input_size=[(B, C, H, W), (B, cond_dim)])