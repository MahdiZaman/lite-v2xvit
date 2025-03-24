import torch
import torch.nn as nn
import torch.nn.functional as F


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        # x: (B, L, 1, 1, 3C)
        batch = x.size(0)
        cav_num = x.size(1)

        if self.radix > 1:
            # x: (B, L, 1, 3, C)
            x = x.view(batch,
                       cav_num,
                       self.cardinality, self.radix, -1)
            x = F.softmax(x, dim=3)
            # B, 3LC
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    def __init__(self, input_dim):
        super(SplitAttn, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim * 3, bias=False)

        self.rsoftmax = RadixSoftmax(3, 1)

    def forward(self, window_list):
        # window list: [(B, L, H, W, C) * 3]
        assert len(window_list) == 3, 'only 3 windows are supported'

        sw, mw, bw = window_list[0], window_list[1], window_list[2]
        B, L = sw.shape[0], sw.shape[1]

        # global average pooling, B, L, H, W, C
        x_gap = sw + mw + bw
        # print(f'sw: {sw.shape}, mw: {mw.shape}, bw: {bw.shape}, x_gap: {x_gap.shape}')        
        x_gap = x_gap.mean((2, 3), keepdim=True)

        x_gap = self.act1(self.bn1(self.fc1(x_gap)))    # B, L, 1, 1, C
        
        x_attn = self.fc2(x_gap)    # # B, L, 1, 1, 3C
        
        x_attn = self.rsoftmax(x_attn).view(B, L, 1, 1, -1)   # B, L, 1, 1, 3C

        out = sw * x_attn[:, :, :, :, 0:self.input_dim] + \
              mw * x_attn[:, :, :, :, self.input_dim:2*self.input_dim] +\
              bw * x_attn[:, :, :, :, self.input_dim*2:]
        return out

def main():
    # Define dummy inputs: three windows with shape [2, 2, 48, 176, 256]
    B, L, H, W, C = 2, 2, 48, 176, 256
    window1 = torch.randn(B, L, H, W, C)
    window2 = torch.randn(B, L, H, W, C)
    window3 = torch.randn(B, L, H, W, C)
    window_list = [window1, window2, window3]

    # Create an instance of SplitAttn with input_dim equal to 256.
    split_attn = SplitAttn(input_dim=C)

    # Run a forward pass through the SplitAttn module.
    output = split_attn(window_list)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()