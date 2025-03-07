import torch
import torch.nn as nn
from torchinfo import summary

class LeNet5(nn.Module):
    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=n_classes)

        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)

        x = self.activation(self.conv2(x))
        x = self.pool(x)

        x = self.activation(self.conv3(x))
        x = torch.flatten(x, start_dim=1)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x
    

if __name__ == "__main__":
    model = LeNet5()
    input = torch.randn(size=(1, 1, 32, 32))
    
    # 모델 구조 출력
    print("\nModel structure:")
    summary(model, input_data=input)