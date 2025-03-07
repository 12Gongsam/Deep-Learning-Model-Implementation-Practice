import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
            nn.Softmax(dim=1), # instead of RBF
        )
        

    def forward(self, x):
        out = self.layers(x)
        return out
    

if __name__ == "__main__":
    model = LeNet5()
    input = torch.randn(size=(1, 1, 32, 32))
    output = model(input)
    print("Output shape:", output.shape)
    
    # 모델 구조 출력
    print("\nModel structure:")
    print(model)
    
    # 전체 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters()) # numel은 텐서의 총 원소 개수를 반환하는 메서드
    print("\nTotal number of parameters:", total_params)