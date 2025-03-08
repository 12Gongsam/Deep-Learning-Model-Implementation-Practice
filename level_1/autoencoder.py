import torch
import torch.nn as nn
from torchinfo import summary

class AutoEncoder(nn.Module):
    def __init__(self, input_dim:int, latent_dims:list):
        super(AutoEncoder, self).__init__()

        self.encoder = self._create_encoder(input_dim, latent_dims)
        self.decoder = self._create_decoder(input_dim, latent_dims)
    
    def _create_encoder(self, input_dim:int, latent_dims:list):
        layers = []
        for in_c, out_c in zip([input_dim] + latent_dims[:-1], latent_dims):
            layers.append(nn.Linear(in_c, out_c))
        return nn.Sequential(*layers)
    
    def _create_decoder(self, input_dim:int, latent_dims:list):
        layers = []
        for in_c, out_c in zip(latent_dims[::-1], latent_dims[::-1][1:] + [input_dim]):
            layers.append(nn.Linear(in_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__=="__main__":
    model = AutoEncoder(
        input_dim=784,
        latent_dims=[1000, 500, 250, 30]
    )
    input = torch.randn(1, 784)
    summary(model, input_data=input)