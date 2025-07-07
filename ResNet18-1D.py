import math
import torch
import torch.nn as nn
import torchinfo


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BasicBlock1D, self).__init__()
        # 수정사항 1. padding 자동 계산
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        # shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # 수정사항 2. ReLU 모듈화
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_main = self.conv1(x)
        out_main = self.bn1(out_main)
        out_main = self.relu(out_main)
        out_main = self.conv2(out_main)
        out_main = self.bn2(out_main)

        out_sc = self.shortcut(x)

        y = self.relu(out_main + out_sc)
        return y


class ResNet18_1D(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(ResNet18_1D, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1 = self._make_stage(64, 64, [1, 1])
        self.stage2 = self._make_stage(64, 128, [2, 1])
        self.stage3 = self._make_stage(128, 256, [2, 1])
        self.stage4 = self._make_stage(256, 512, [2, 1])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes)
        )
        self.apply(self._initialize_weights)

    def _make_stage(self, in_channels, out_channels, strides: list):
        layers = []
        for s in strides:
            layers.append(
                BasicBlock1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=s
                )
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.head(x)
        return x


if __name__ == "__main__":
    B, C, N = 1, 1, 16000
    # torch.randn() 평균 0 분산 1인 정규분포에서 샘플링한 값으로 텐서를 생성
    dummy_input = torch.randn(size=(B, C, N))
    
    model = ResNet18_1D(10, C)
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"Output shape : {output.shape}")
        except Exception as e:
            print(e)
    torchinfo.summary(model, input_size=(B, C, N))