import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from typing_extensions import Protocol
from typing import Sequence, Tuple, Union


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import math
import time

# import ptwt
import pywt
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse



class GlobalAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        """
        Global self-attention module for feature maps.
        Args:
            dim (int): Embedding dimension (channel dimension).
            heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super(GlobalAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # print(f'x input to Global Attention: {x.shape}')
        
        # x is expected to have shape [B, L, H, W, C]
        B, L, H, W, C = x.shape
        # Flatten spatial dimensions: shape -> [B*L, H*W, C]
        x_flat = x.view(B * L, H * W, C)
        # print(f'x_flat: {x_flat.shape}')
        
        # Apply global multi-head attention
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        # print(f'attn_out: {attn_out.shape}')
        x_residual = self.norm1(x_flat + attn_out)
        # print(f'x_residual: {x_residual.shape}')
        
        # Feed-forward network (MLP) with residual connection
        mlp_out = self.mlp(x_residual)
        # print(f'mlp_out: {mlp_out.shape}')
        x_out = self.norm2(x_residual + mlp_out)
        # print(f'x_out: {x_out.shape}')
        
        # Reshape back to [B, L, H, W, C]
        x_out = x_out.view(B, L, H, W, C)
        # print(f'x_out reshaped: {x_out.shape}')
        return x_out


def get_relative_distances(window_size):
    indices = torch.tensor(np.array(
        [[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class BaseWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, drop_out, window_size,
                 relative_pos_embedding):
        super().__init__()
        inner_dim = dim_head * heads
        # print(dim, heads, dim_head, drop_out, window_size,
        #          relative_pos_embedding)
        # print(inner_dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + \
                                    window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1,
                                                          2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2,
                                                          window_size ** 2))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        b, l, h, w, c, m = *x.shape, self.heads

        # print(f'x input to Base Attention: {x.shape}')
        qkv = self.to_qkv(x).chunk(3, dim=-1)   # chunk returns a list
        # print(f'qkv: {qkv[0].shape}, {qkv[1].shape}, {qkv[2].shape}')
        new_h = h // self.window_size
        new_w = w // self.window_size

        # q : (b, l, m, new_h*new_w, window_size^2, c_head)
        w_h = self.window_size
        w_w = self.window_size
        
        # print(f'h: {h}, w: {w}, new_h: {new_h}, new_w: {new_w}, m: {m}, c: {c}')
        # print(f'w_h: {w_h}, w_w: {w_w}')
        q, k, v = map(
            lambda t: rearrange(t,
                                'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c',
                                m=m, w_h=self.window_size,
                                w_w=self.window_size), qkv)
        # print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')
        
        # b l m h window_size window_size
        dots = torch.einsum('b l m h i c, b l m h j c -> b l m h i j',
                            q, k, ) * self.scale
        # consider prior knowledge of the local window
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0],
                                       self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)
        # print(f'attn: {attn.shape}')

        out = torch.einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)
        # print(f'out1: {out.shape}')
        
        # b l h w c
        out = rearrange(out,
                        'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)',
                        m=self.heads, w_h=self.window_size,
                        w_w=self.window_size,
                        new_w=new_w, new_h=new_h)
        # print(f'out2: {out.shape}')
        
        out = self.to_out(out)
        # print(f'out3: {out.shape}')

        return out


class WaveletTransform2D(torch.nn.Module):
    def __init__(self, wavelet='db1', level=1, mode='zero'):
        super(WaveletTransform2D, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.dwt = DWTForward(J=level, wave=wavelet, mode=mode)

    def forward(self, x):
        # For example, if x has shape [B, L, H, W, C], first bring the channels to the proper position.
        # print(f'input x: {x.shape}')
        x = x.permute(0, 1, 4, 2, 3)  # New shape: [B, L, C, H, W]
        # print(f'permuted x: {x.shape}')
           
        # Merge batch and length dimensions so that we have a 4D tensor [B*L, C, H, W]
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        
        # Apply the wavelet transform on the entire tensor on the current device
        Yl, Yh = self.dwt(x)
        # print(f'Yl: {Yl.shape}')
        
        # Optionally, reshape the results back to include the original batch and sequence dimensions.
        Yl = Yl.reshape(B, L, *Yl.shape[1:]).permute(0, 1, 3, 4, 2)
        
        ## FIXME: Yh is untreated. Fix before using IDWT.  
        # Yh = [detail.reshape(B, L, *detail.shape[1:]) for detail in Yh]
        
        # print(f'Yl: {Yl.shape}')
        return Yl, Yh


class InverseWaveletTransform2D(nn.Module):
    def __init__(self, wavelet='db1', level=1, mode='zero'):
        """
        Args:
            wavelet (str): Name of the wavelet to use.
            level (int): Decomposition level.
            mode (str): Signal extension mode (e.g., 'zero', 'symmetric', etc.).
        """
        super(InverseWaveletTransform2D, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.idwt = DWTInverse(wave=wavelet, mode=mode)

    def forward(self, Yl, Yh):
        """
        Reconstruct the original 2D signal from the approximation coefficients Yl
        and the detail coefficients Yh.

        Args:
            Yl (torch.Tensor): Approximation coefficients with shape [B, L, H_low, W_low, C].
            Yh (list): List of detail coefficients for each decomposition level as produced by DWTForward.
        
        Returns:
            rec_tensor (torch.Tensor): Reconstructed tensor with shape [B, L, H, W, C].
        """
        B, L, H_low, W_low, C = Yl.shape

        # Permute Yl back to [B, L, C, H_low, W_low]
        Yl = Yl.permute(0, 1, 4, 2, 3).contiguous()
        # Merge batch and sequence dimensions: [B*L, C, H_low, W_low]
        Yl = Yl.view(B * L, C, H_low, W_low)
        
        # Apply the inverse wavelet transform.
        # It expects the tuple (Yl, Yh) with Yl of shape [B*L, C, H_low, W_low].
        rec = self.idwt((Yl, Yh))  # rec shape: [B*L, C, H, W]
        
        # Reshape back to [B, L, C, H, W] then permute to [B, L, H, W, C]
        rec = rec.view(B, L, C, rec.shape[2], rec.shape[3])
        rec_tensor = rec.permute(0, 1, 3, 4, 2).contiguous()
        return rec_tensor

    

class WaveletWindowAttention(nn.Module):
    def __init__(self, wavelet='db1', level=1, mode='zero',
                 dim=64, heads=4, dim_head=16, drop_out=0.1, window_size=7, relative_pos_embedding=True):
        """
        Combined model that applies a wavelet transform followed by a window attention module.
        
        Args:
            wavelet (str): Type of wavelet for the transform.
            level (int): Number of decomposition levels.
            mode (str): Signal extension mode.
            dim (int): Input channel dimension for the attention module.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            drop_out (float): Dropout rate in the attention module.
            window_size (int): Spatial window size for local attention.
            relative_pos_embedding (bool): Whether to use relative positional embeddings.
        """
        super(WaveletWindowAttention, self).__init__()
        
        self.wavelet_transform = WaveletTransform2D(wavelet=wavelet, level=level, mode=mode)
        
        # self.window_attention = BaseWindowAttention(dim=dim, heads=heads, dim_head=dim_head,
        #                                             drop_out=drop_out, window_size=window_size,
        #                                             relative_pos_embedding=relative_pos_embedding)
        self.global_attention = GlobalAttention(dim=dim, heads=heads, dropout=0.3)
        self.inverse_wavelet_transform = InverseWaveletTransform2D(wavelet=wavelet, level=level, mode=mode)
        

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, H, W, C].
            
        Returns:
            attn_out (torch.Tensor): Output tensor from the attention module.
            Yh (list): List of detail coefficients from the wavelet transform.
        """
        # [B, L, 48, 176, 256]
        
        # This returns Yl (approximation coefficients) and Yh (detail coefficients).
        Yl, Yh = self.wavelet_transform(x)  # Yl -> [B, L, 3, 11, 256]
        
        # Pass the low-frequency approximation coefficients to the attention module.
        # Yl is expected to be in the shape [B, L, H', W', C'].
        # print(f'Yl input to window attention: {Yl.shape}')
        
        # TODO mzaman : FIXME BaseWindowAttention is not adapted to wavelet downsampled input
        # attn_out = self.window_attention(Yl)
        
        # Using Global Attention on the smallest spatial scale. 
        attn_out = self.global_attention(Yl)
        # print(f'attn_out: {attn_out.shape}')
        
        reconstructed = self.inverse_wavelet_transform(attn_out, Yh)
        # print(f'reconstructed: {reconstructed.shape}')        
        
        # return attn_out, Yh
        return reconstructed

        
def main():
    # Define dummy input parameters
    B = 2  # Batch size
    L= 3
    H = 48  # Height of the image
    W = 176  # Width of the image
    C = 256  # Number of channels
    
    
    # Create a dummy input tensor
    dummy_input = torch.randn(B, L, H, W, C)

    
    # # WaveletTransform3D
    # wavelet_transform = WaveletTransform2D(wavelet='db1', level=1, mode='zero')
    # out_l, out_h = wavelet_transform(dummy_input)
    
    model = WaveletWindowAttention(
        wavelet='db1',
        level=4,
        mode='zero',
        dim=C,
        heads=4,
        dim_head=16,
        drop_out=0.1,
        window_size=4,
        relative_pos_embedding=True
    )
    print(f'input: {dummy_input.shape}')
    output = model(dummy_input)
    print(f'output: {output.shape}')
    
if __name__ == "__main__":
    main()