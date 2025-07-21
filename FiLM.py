"""
Feature-wise Linear Modulation (FiLM) Implementation
====================================================

This module provides skeleton classes for building FiLM-conditioned
networks as described in:

    Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville,  
    “FiLM: Visual Reasoning with a General Conditioning Layer,”  
    *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence
    (AAAI-18)*, New Orleans, LA, USA, 2018.

Paper available at:
https://arxiv.org/abs/1709.07871
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
print("import done")

class FiLMLayer(nn.Module):
    """
    One feature-wise linear modulation layer.

    Parameters
    ----------
    num_features : int
        Number of feature maps to modulate.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D401
        """Apply FiLM: `gamma * x + beta`."""
        B, C, *_ = x.shape

        # ---- Channel-dimension checks -----------------------------------
        if C != self.num_features:
            raise ValueError(
                f"[FiLMLayer] Expected x to have {self.num_features} channels, "
                f"but got {C}."
            )
        if gamma.shape != (B, C) or beta.shape != (B, C):
            raise ValueError(
                "[FiLMLayer] gamma / beta must both be (B, C): "
                f"gamma {tuple(gamma.shape)}, beta {tuple(beta.shape)}, "
                f"x batch {B}."
            )
        return gamma[..., None, None] * x + beta[..., None, None]


class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (γ, β) from a conditioning vector.

    Typical choice: a GRU that encodes the question or any other
    modality and projects to 2 × num_features.
    """

    def __init__(self, vocab_size, num_blocks=4, num_channels=128,
                 emb_dim=200, hidden=4096):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.embed = nn.Embedding(vocab_size, embedding_dim=emb_dim)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.to_gamma_beta = nn.Linear(
            hidden,
            num_blocks * 2 * num_channels
        )

    def forward(self, tokens): # B, T
        """
        Parameters
        ----------
        cond : torch.Tensor
            Conditioning input, e.g. word embeddings of shape (B, T, E).

        Returns
        -------
        gamma, beta : torch.Tensor
            FiLM parameters of shape (B, C).
        """
        B = tokens.size(0)
        x = self.embed(tokens) # B, T, E
        _, h_n = self.gru(x) # 1, B, H
        params = self.to_gamma_beta(h_n.squeeze(0))
        params = params.view(B, self.num_blocks, 2, self.num_channels)
        gamma, beta = params[:, :, 0, :], params[:, :, 1, :]
        return list(gamma.unbind(1)), list(beta.unbind(1))


class FiLMResidualBlock(nn.Module):
    """
    Residual block with internal FiLM modulation.

    Structure
    ---------
    conv1 → norm1 → FiLM → ReLU → conv2 → norm2 → FiLM → ReLU → residual add
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # define two convolutions and normalizations here
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film_layer = FiLMLayer(channels)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channels)

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation twice inside the residual block.

        The same (γ, β) can be reused or split for each conv layer.
        """
        x = self.conv1(x)
        x = self.relu(x)
        out = self.conv2(x)
        out = self.bn(out)
        out = self.film_layer(out, gamma, beta)
        out = self.relu(out)
        return x + out


class FiLMResNet(nn.Module):
    """
    Simple CNN backbone modulated by FiLM at multiple depths.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB).
    num_blocks : int
        Number of FiLMResidualBlocks to stack.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        num_classes: int = 28
    ) -> None:
        super().__init__()
        # stem convolution
        # stack of FiLMResidualBlocks
        # global pooling + classifier
        resnet101 = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V2
        )
        self.backbone = nn.Sequential(
            resnet101.conv1, resnet101.bn1, resnet101.relu, resnet101.maxpool,
            resnet101.layer1, resnet101.layer2, resnet101.layer3
        )
        # 가중치 동결
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        self.reducer = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock(hidden_channels)
                for _ in range(num_blocks)
            ]
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, 512, kernel_size=1),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(1),
            nn.Linear(512, 1024),
            nn.Linear(1024, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        gamma_list: list[torch.Tensor],
        beta_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through the FiLM-conditioned CNN.

        gamma_list / beta_list should contain one entry per FiLM block.
        """
        features = self.backbone(x) # (B,1024,14,14)
        features = self.reducer(features) # (B,128,14,14)
        for i, resblock in enumerate(self.blocks):
            features = resblock(features, gamma_list[i], beta_list[i])
        
        logits = self.head(features)
        return logits


if __name__=="__main__":
    B, T = 16, 100
    dummy_tokens = torch.randint(low=1, high=100, size=(B, T))

    C, H, W = 3, 224, 224
    dummy_images = torch.randn(B, C, H, W)
    
    film_generator = FiLMGenerator(1000)
    model = FiLMResNet()

    gamma_list, beta_list = film_generator(dummy_tokens)
    logits = model(dummy_images, gamma_list, beta_list)

    print(logits.shape)