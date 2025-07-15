"""
PixelCNN Implementation

This module implements PixelRNN as described in:

    Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu,
    “Pixel Recurrent Neural Networks,” in Proceedings of the
    33rd International Conference on Machine Learning (ICML),
    New York, NY, USA, 2016.

Paper available at:
https://arxiv.org/abs/1601.06759
"""

from enum import Enum
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo


class MaskType(Enum):
    """
    Enum to distinguish between the two convolution masks used in PixelCNN.

    - A: 첫 합성곱층에 사용. 현재 픽셀(채널 포함)을 모두 마스킹.
    - B: 나머지 모든 합성곱층에 사용. 현재 픽셀의 과거 채널(R→G→B 순)을 살림.
    """
    A = "A"
    B = "B"

    def is_first_layer(self) -> bool:
        return self is MaskType.A


class MaskedConv2d(nn.Conv2d):
    """
    2-D masked convolution layer.

    Parameters
    ----------
    in_channels : int
        입력 피처 채널 수.
    out_channels : int
        출력 피처 채널 수.
    kernel_size : int
        커널 한 변의 길이(정사각 형태).
    mask_type : MaskType
        A 또는 B 마스크 선택.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask_type: MaskType,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.mask_type = mask_type
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self._build_mask()

    def _build_mask(self) -> None:
        """
        마스킹 행렬(self.mask)을 생성합니다.
        Mask A/B 규칙에 따라 현재 픽셀·채널 가중치를 0으로 설정하세요.
        """
        # mask를 모두 1로 초기화
        self.mask.data.fill_(1)
        # kernel 크기에서 중앙 좌표 계산
        _, _, kH, kW = self.weight.shape
        cy, cx = kH // 2, kW // 2
        # 중앙 행 아래쪽 전부 0
        self.mask[:, :, cy + 1:, :] = 0
        # 중앙 행에서 중앙 열 오른쪽 전부 0
        self.mask[:, :, cy, cx + 1:] = 0
        # Mask A 인 경우 중앙 위치도 0으로
        if self.mask_type is MaskType.A:
            self.mask[:, :, cy, cx] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (1) self.weight ∘ self.mask 로 마스킹 연산을 수행한 후
        (2) 표준 Conv2d forward 를 호출합니다.
        """
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


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = MaskedConv2d(
            in_channels=channels,
            out_channels=channels//2,
            kernel_size=1,
            mask_type=MaskType.B,
        )
        self.conv2 = MaskedConv2d(
            in_channels=channels//2,
            out_channels=channels//2,
            kernel_size=3,
            mask_type=MaskType.B,
        )
        self.conv3 = MaskedConv2d(
            in_channels=channels//2,
            out_channels=channels,
            kernel_size=1,
            mask_type=MaskType.B,
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual skip connection.
        """
        out = self.conv1(self.relu(x))
        out = self.conv2(self.relu(out))
        out = self.conv3(self.relu(out))
        return x + out


class PixelCNN(nn.Module):
    """
    Complete PixelCNN network.

    Parameters
    ----------
    in_channels : int
        입력 채널(RGB=3).
    hidden_channels : int
        첫 conv 및 잔차 블록 피처 맵 채널 수(h).
    num_residual_blocks : int
        Residual block 반복 횟수(L).
    num_output_classes : int, default 256
        픽셀당 클래스 개수(8-bit → 256).
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 256,
        num_residual_blocks: int = 15,
        num_output_classes: int = 256,
    ) -> None:
        super().__init__()

        # 첫 Mask-A 합성곱(7×7)
        self.conv0 = MaskedConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=7,
            mask_type=MaskType.A,
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_channels)
                for _ in range(num_residual_blocks)
            ]
        )

        # 두 개의 1×1 projection 후 로짓 출력
        self.project1 = MaskedConv2d(
            in_channels=hidden_channels,
            out_channels=1024,
            kernel_size=1,
            mask_type=MaskType.B,
        )
        self.project2 = MaskedConv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=1,
            mask_type=MaskType.B,
        )
        self.classifier = nn.Conv2d(
            in_channels=1024,
            out_channels=in_channels * num_output_classes,
            kernel_size=1,
        )
        self.relu = nn.ReLU()
        self.in_channels = in_channels
        self.num_output_classes = num_output_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W) float32, 입력 이미지 [0, 1] 또는 [-1, 1].

        Returns
        -------
        logits : (B, C, 256, H, W)
            픽셀당 클래스 로짓. 채널별로 256-way Softmax를 적용하면 확률 분포가 됩니다.
        """
        x = self.conv0(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.project1(self.relu(x))
        x = self.project2(self.relu(x))
        logits = self.classifier(x)
        logits = logits.unflatten(1, (self.in_channels, self.num_output_classes))
        return logits
        

    def generate(
        self,
        shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        오토리그레시브 샘플링 루프.

        Parameters
        ----------
        shape : (B, C, H, W)
            생성할 이미지 텐서 형태.
        device : torch.device, optional
            결과 텐서를 올릴 장치.
        temperature : float
            Softmax temperature (>0).

        Returns
        -------
        samples : (B, C, H, W) uint8
            샘플링된 이미지.
        """
        if device is None:
            device = next(self.parameters()).device
        
        B, C, H, W = shape
        assert C == self.in_channels

        samples = torch.zeros(shape, dtype=torch.uint8, device=device)

        # 이 구현은 픽셀 단위 루프라서 Python 오버헤드가 큽니다(32×32×3 ≈ 3 k forward 호출).
        # 배치 × GPU 환경에서는 64×64 이미지를 생성해도 수 초면 충분하지만, 
        # 큰 해상도(>128²)에서는 블록 샘플링 또는 캐시 기법(무브먼트 프리루프)을 고려하세요
        self.eval()
        with torch.no_grad():
            for y in range(H):
                for x in range(W):
                    for c in range(C):
                        # samples는 uint8(0-255) 이므로 float으로 바꾼뒤 255로 나눠
                        # [0, 1] 범위로 스케일링
                        inp = samples.float() / 255.0
                        logits = self.forward(inp)
                        logits = logits[:, c, :, y, x]

                        logits = logits / temperature
                        probs = F.softmax(logits, dim=-1) # B, 256

                        vals = torch.multinomial(probs, num_samples=1)
                        samples[:, c, y, x] = vals.squeeze(-1).to(torch.uint8)
        return samples


if __name__=="__main__":
    dummy_input = torch.randn(1, 3, 32, 32)
    model = PixelCNN()
    output = model(dummy_input)
    print(output.shape)
    torchinfo.summary(model, input_size=(1, 3, 32, 32))