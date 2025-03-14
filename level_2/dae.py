import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class DAE(nn.Module):
    def __init__(self, in_features, out_features):
        super(DAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_features, in_features),
            nn.Sigmoid()
        )

    def _add_noise(self, x, noise_ratio=0.1):
        noisy_x = x.clone()  # 입력 데이터 복사
        batch_size, num_features = noisy_x.shape  # (batch_size, feature_dim)
        
        num_noisy = int(num_features * noise_ratio)  # 한 샘플당 손상시킬 요소 개수
        
        for i in range(batch_size):
            noisy_idx = torch.randperm(num_features)[:num_noisy]  # 각 샘플마다 다른 인덱스 선택
            noisy_x[i, noisy_idx] = 0  # 해당 샘플의 특정 인덱스를 0으로 설정
        
        return noisy_x

    def forward(self, x):
        x = self._add_noise(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

