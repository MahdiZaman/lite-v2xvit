import torch
import torch.nn as nn


class NaiveCompressor(nn.Module):
    """
    A very naive compression that only compress on the channel.
    """
    def __init__(self, input_dim, compress_raito):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//compress_raito, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim//compress_raito, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim//compress_raito, input_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3,
                           momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    

if __name__ == '__main__':
    input_dim = 256
    compress_raito = 2
    x = torch.randn(1, input_dim, 64, 64)
    print(x.size())
    naive_compressor = NaiveCompressor(input_dim, compress_raito)
    out = naive_compressor(x)
    print(out.size())