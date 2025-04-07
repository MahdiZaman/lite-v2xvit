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
    def __init__(self, dim, heads, dim_head, drop_out, window_size, relative_pos_embedding):
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




def get_relative_distances_rect(window_height, window_width):
    # Create a list of [i, j] coordinates for a grid of size (window_height, window_width)
    indices = torch.tensor([[i, j] for i in range(window_height) for j in range(window_width)])
    # print(f'indices: {indices.shape}')
    
    # Compute pairwise differences
    relative_indices = indices[None, :, :] - indices[:, None, :]  # shape: [window_height*window_width, window_height*window_width, 2]
    # print(f'relative_indices: {relative_indices.shape}')
    
    # Shift the differences so they are non-negative
    relative_indices[..., 0] += window_height - 1
    relative_indices[..., 1] += window_width - 1
    
    return relative_indices


class RectWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, drop_out, window_size,
                 relative_pos_embedding):
        """
        BaseWindowAttention Adapted for a rectangular window. 
        `window_size` should now be a tuple: (window_height, window_width)
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # Change 1: Accept a tuple for window size.
        if isinstance(window_size, (list, tuple)):
            self.window_height = window_size[0]
            self.window_width = window_size[1]
        else:
            self.window_height = window_size
            self.window_width = window_size
        
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # print(f'dim: {dim}, inner_dim: {inner_dim}')
        # exit()

        # Change 2: Adapt relative positional encoding for rectangular windows.
        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances_rect(self.window_height, self.window_width)
            self.pos_embedding = nn.Parameter(torch.randn(2 * self.window_height - 1,
                                                          2 * self.window_width - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(self.window_height * self.window_width,
                                                          self.window_height * self.window_width))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        # Here x is expected to be of shape [B, L, H, W, C]
        # In our use case: [B, L, 3, 11, 256]
        b, l, h, w, c = x.shape  # h should be 3, w should be 11
        m = self.heads  # Here, m = 4
        
        #print(f'x input to Base Attention: {x.shape}')
        #print(f'b: {b}, l: {l}, h: {h}, w: {w}, c: {c}, m: {m}')
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        #print(f'qkv: {qkv[0].shape}, {qkv[1].shape}, {qkv[2].shape}')
        
        # Change 3: Use rectangular window sizes.
        # Compute number of windows along each spatial dimension.
        new_h = h // self.window_height  # for input, 3//3 = 1
        new_w = w // self.window_width   # for input, 11//11 = 1

        w_h = self.window_height
        w_w = self.window_width
        
        #print(f'h: {h}, w: {w}, new_h: {new_h}, new_w: {new_w}, m: {m}, c: {c}')
        #print(f'w_h: {w_h}, w_w: {w_w}')
        
        # Change 4: Update rearrange to use window_height and window_width.
        q, k, v = map(
            lambda t: rearrange(t,
                                'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c',
                                m=m, w_h=self.window_height, w_w=self.window_width),
            qkv)
        #print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')
        
        # Compute attention scores with scaled dot-product and modulate by relative position bias.
        dots = torch.einsum('b l m h i c, b l m h j c -> b l m h i j',
                            q, k) * self.scale
        #print(f'dots: {dots.shape}')
        
        if self.relative_pos_embedding:
            # Change 5: The indexing remains the same since self.relative_indices is computed for a rectangular window.
            dots += self.pos_embedding[self.relative_indices[:, :, 0],
                                       self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)
        #print(f'attn: {attn.shape}')

        out = torch.einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)
        #print(f'out1: {out.shape}')
        
        # Change 6: Rearranging using the updated window dimensions.
        out = rearrange(out,
                        'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)',
                        m=self.heads, w_h=self.window_height, w_w=self.window_width,
                        new_w=new_w, new_h=new_h)
        #print(f'out2: {out.shape}')
        
        out = self.to_out(out)
        #print(f'out3: {out.shape}')

        return out
    

class WaveletWindowAttentionSingleScale(nn.Module):
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
        
        
        self.window_attention = RectWindowAttention(dim=dim, heads=heads, dim_head=dim_head,
                                                    drop_out=drop_out, window_size=window_size,
                                                    relative_pos_embedding=relative_pos_embedding)
        # self.global_attention = GlobalAttention(dim=dim, heads=heads, dropout=0.3)
        
        self.wavelet_transform = WaveletTransform2D(wavelet=wavelet, level=level, mode=mode)
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
        print(f'wavelet input: {x.shape}')
        
        # This returns Yl (approximation coefficients) and Yh (detail coefficients).
        Yl, Yh = self.wavelet_transform(x)  # Yl -> [B, L, 3, 11, 256]
        
        # Pass the low-frequency approximation coefficients to the attention module.
        # Yl is expected to be in the shape [B, L, H', W', C'].
        print(f'Yl input to window attention: {Yl.shape}')
        
        attn_out = self.window_attention(Yl)    
        # attn_out = self.global_attention(Yl)
        print(f'attn_out: {attn_out.shape}')
        
        reconstructed = self.inverse_wavelet_transform(attn_out, Yh)
        print(f'reconstructed: {reconstructed.shape}')        
        
        # return attn_out, Yh
        return reconstructed


class WaveletWindowAttention(nn.Module):
    def __init__(self, wavelet='db1', level=[1,2], mode='zero',
                 dim=64, heads=4, dim_head=16, drop_out=0.1, window_size=7, relative_pos_embedding=True):
        """
        Combined model that applies a wavelet transform followed by a window attention module.
        Now supports multiple decomposition levels provided as a list of integers.
        
        Args:
            wavelet (str): Type of wavelet for the transform.
            levels (list of int): List of decomposition levels.
            mode (str): Signal extension mode.
            dim (int): Input channel dimension for the attention module.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            drop_out (float): Dropout rate in the attention module.
            window_size (int or tuple): Spatial window size for local attention.
            relative_pos_embedding (bool): Whether to use relative positional embeddings.
        """
        super(WaveletWindowAttention, self).__init__()
        
        # The same window attention module is used for each level.
        self.window_attention = RectWindowAttention(dim=dim, heads=heads, dim_head=dim_head,
                                                    drop_out=drop_out, window_size=window_size,
                                                    relative_pos_embedding=relative_pos_embedding)
        # self.global_attention = GlobalAttention(dim=dim, heads=heads, dropout=0.3)
        
        # CHANGES:
        # Instead of a single wavelet transform, create a module list for each level.
        self.wavelet_transforms = nn.ModuleList([
            WaveletTransform2D(wavelet=wavelet, level=lev, mode=mode) for lev in level
        ])
        # Create a corresponding module list for the inverse transforms.
        self.inverse_wavelet_transforms = nn.ModuleList([
            InverseWaveletTransform2D(wavelet=wavelet, level=lev, mode=mode) for lev in level
        ])
        self.level = level  # store levels for reference

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, H, W, C].
            
        Returns:
            merged_reconstruction (torch.Tensor): Merged reconstruction from all levels.
                Typically, shape [B, L, H, W, C].
        """
        # CHANGES:
        # Run each wavelet transform and process its approximation with window attention.
        reconstructions = []  # Collect the reconstructed outputs for each level.
        for wavelet_transform, inverse_wavelet_transform in zip(self.wavelet_transforms, self.inverse_wavelet_transforms):
            #print(f'wavelet input---------------------------------------------: {x.shape}')
            # Wavelet transform: returns approximation (Yl) and detail coefficients (Yh)
            Yl, Yh = wavelet_transform(x)
            #print(f'Yl input to window attention: {Yl.shape}')
            # Apply window attention on the low-frequency approximation.
            attn_out = self.window_attention(Yl)
            #print(f'attn_out: {attn_out.shape}')
            # Inverse wavelet transform: reconstructs the full-resolution output.
            rec = inverse_wavelet_transform(attn_out, Yh)
            #print(f'reconstructed: {rec.shape}')
            reconstructions.append(rec)
        
        # Merge the reconstructions from different levels.
        # For example, one can take an element-wise average.
        merged_reconstruction = sum(reconstructions) / len(reconstructions)
        #print(f'merged_reconstruction: {merged_reconstruction.shape}')
        return merged_reconstruction

        
def main():
    # Define dummy input parameters
    B = 2  # Batch size
    L= 3
    H = 48  # Height of the image
    W = 176  # Width of the image
    C = 256  # Number of channels
    
    
    # Create a dummy input tensor
    dummy_input = torch.randn(B, L, H, W, C)
    
    model = WaveletWindowAttention(
        wavelet='db1',
        level=[2,3,4],
        mode='zero',
        dim=C,
        heads=4,
        dim_head=64,
        drop_out=0.1,
        window_size=(3,11),  # Rectangular window size
        relative_pos_embedding=True
    )
    print(f'input: {dummy_input.shape}')
    output = model(dummy_input)
    print(f'output: {output.shape}')
    
if __name__ == "__main__":
    main()