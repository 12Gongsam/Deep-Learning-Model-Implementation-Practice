import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = self._create_conv_layer(3, 96, 11, 
                                             stride=4, padding=2, use_lrn=True, use_maxpool=True)
        self.conv2 = self._create_conv_layer(96, 256, 5,
                                             stride=1, padding=2, use_lrn=True, use_maxpool=True)
        self.conv3 = self._create_conv_layer(256, 384, 3,
                                             stride=1, padding=1)
        self.conv4 = self._create_conv_layer(384, 384, 3,
                                             stride=1, padding=1)
        self.conv5 = self._create_conv_layer(384, 256, 3,
                                             stride=1, padding=1, use_maxpool=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        )

    def _create_conv_layer(self, in_channels, out_channels, kernel_size, 
                           stride=1, padding=0, use_lrn=False, use_maxpool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU()]
        
        if use_lrn:
            layers.append(nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0))
            
        if use_maxpool:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


if __name__=="__main__":
    model = AlexNet()
    input = torch.randn(size=(1, 3, 224, 224))
    
    # 모델 구조 출력
    print("\nModel structure:")
    summary(model, input_data=input)