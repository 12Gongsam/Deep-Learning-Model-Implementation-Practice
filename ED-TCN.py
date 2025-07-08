"""
Temporal Convolutional Network (ED-TCN) Implementation

This module implements the Encoder–Decoder TCN (ED-TCN) as described in:

    C. Lea, M. Flynn, R. Vidal, A. Reiter, G. D. Hager,
    “Temporal Convolutional Networks: A Unified Approach to Action Segmentation,”
    in European Conference on Computer Vision (ECCV), 2016.

You can find the paper at:
https://arxiv.org/abs/1603.09440
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo


class ChannelMaxNorm(nn.Module):
    """채널별 최대값으로 정규화하는 레이어.

    Notes
    -----
    입력 텐서 (B, C, T)에서 시점별 채널 최대값으로 나누어
    조명·센서 스케일 변화에 대한 강인성을 부여합니다.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, eps: float = 1e-5):
        """정규화 연산을 수행합니다.

        Parameters
        ----------
        x : torch.Tensor
            입력 텐서 (B, C, T).
        eps : float, optional
            0 나눗셈 방지를 위한 작은 값.
        """
        m = torch.amax(x, dim=1, keepdim=True)
        # 수정사항 1. eps dtype, device를 맞추어서 불필요한 casting 없애기
        m = torch.clamp(m, min=torch.tensor(eps, dtype=x.dtype, device=x.device))
        normalized_m = x / m
        return normalized_m



class TemporalConvEncoderLayer(nn.Module):
    """Encoder 단일 층.

    구성
    ----
    Conv1d → LeakyReLU → MaxPool1d(2) → ChannelMaxNorm
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True
        )
        self.act = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.normalization = ChannelMaxNorm()

    def forward(self, x):
        """한 층의 전방 계산을 수행합니다."""
        out = self.conv(x)
        out = self.act(out)
        out = self.maxpool(out)
        out = self.normalization(out)
        return out


class TemporalConvDecoderLayer(nn.Module):
    """Decoder 단일 층.

    구성
    ----
    Upsample(2) → Conv1d → LeakyReLU → ChannelMaxNorm
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True
        )
        self.act = nn.LeakyReLU(inplace=True)
        self.normalization = ChannelMaxNorm()

    def forward(self, x):
        """한 층의 전방 계산을 수행합니다."""
        out = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.conv(out)
        out = self.act(out)
        out = self.normalization(out)
        return out


class TemporalConvNet(nn.Module):
    """Encoder-Decoder 형태의 TCN 전체 모델.

    Parameters
    ----------
    input_channels : int
        입력 특성 차원 F₀.
    num_classes : int
        예측할 클래스 수.
    filter_channels : tuple[int], optional
        각 Encoder 층의 출력 채널 수.
    kernel_size : int, optional
        Conv1d 필터 길이.
    """
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        filter_channels: tuple[int] = (32, 64, 96),
        kernel_size: int = 5,
    ):
        super().__init__()
        # encoder 구성
        channels = (input_channels, *filter_channels)
        self.encoder_channels = list(zip(channels, channels[1:]))
        encoder_layers = []
        for (in_c, out_c) in self.encoder_channels:
            encoder_layers.append(
                TemporalConvEncoderLayer(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder 구성
        self.decoder_channels = [
            (out_c, in_c)
            for (in_c, out_c) in reversed(self.encoder_channels)
        ]
        decoder_layers = []
        for (in_c, out_c) in self.decoder_channels:
            decoder_layers.append(
                TemporalConvDecoderLayer(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size
                )
            )
        self.decoder = nn.Sequential(*decoder_layers)

        # head 정의
        self.head = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=True
        )


    def forward(self, x):
        """모델 전방 계산.

        Parameters
        ----------
        x : torch.Tensor
            입력 텐서 (B, F₀, T).

        Returns
        -------
        torch.Tensor
            시점별 로짓 (B, F₀, T).
        """
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.head(out)
        return out


if __name__=="__main__":
    B, C, T = 1, 3, 16000
    # torch.randn() 평균 0 분산 1인 정규분포에서 샘플링한 값으로 텐서를 생성
    dummy_input = torch.randn(size=(B, C, T))
    
    model = TemporalConvNet(
        input_channels=C,
        num_classes=10,
    )
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"Output shape : {output.shape}")
        except Exception as e:
            print(e)
    torchinfo.summary(model, input_size=(B, C, T))