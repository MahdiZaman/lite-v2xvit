"""
Multi-scale window transformer
"""
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from opencood.models.sub_modules.split_attn import SplitAttn


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

        print(f'x input to Base Attention: {x.shape}')
        qkv = self.to_qkv(x).chunk(3, dim=-1)   # chunk returns a list
        print(f'qkv: {qkv[0].shape}, {qkv[1].shape}, {qkv[2].shape}')
        new_h = h // self.window_size
        new_w = w // self.window_size

        # q : (b, l, m, new_h*new_w, window_size^2, c_head)
        w_h = self.window_size
        w_w = self.window_size
        
        print(f'h: {h}, w: {w}, new_h: {new_h}, new_w: {new_w}, m: {m}, c: {c}')
        print(f'w_h: {w_h}, w_w: {w_w}')
        q, k, v = map(
            lambda t: rearrange(t,
                                'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c',
                                m=m, w_h=self.window_size,
                                w_w=self.window_size), qkv)
        print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')
        
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
        print(f'attn: {attn.shape}')

        out = torch.einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)
        print(f'out1: {out.shape}')
        
        # b l h w c
        out = rearrange(out,
                        'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)',
                        m=self.heads, w_h=self.window_size,
                        w_w=self.window_size,
                        new_w=new_w, new_h=new_h)
        print(f'out2: {out.shape}')
        
        out = self.to_out(out)
        print(f'out3: {out.shape}')

        return out


class PyramidWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads, drop_out, window_size,
                 relative_pos_embedding, fuse_method='naive'):
        super().__init__()

        assert isinstance(window_size, list)
        assert isinstance(heads, list)
        assert isinstance(dim_heads, list)
        assert len(dim_heads) == len(heads)

        self.pwmsa = nn.ModuleList([])

        for (head, dim_head, ws) in zip(heads, dim_heads, window_size):
            self.pwmsa.append(BaseWindowAttention(dim,
                                                  head,
                                                  dim_head,
                                                  drop_out,
                                                  ws,
                                                  relative_pos_embedding))
        self.fuse_mehod = fuse_method
        if fuse_method == 'split_attn':
            self.split_attn = SplitAttn(256)

    def forward(self, x):
        print(f'x input to PyramidAttention: {x.shape}')
        output = None
        # naive fusion will just sum up all window attention output and do a
        # mean
        if self.fuse_mehod == 'naive':
            for wmsa in self.pwmsa:
                output = wmsa(x) if output is None else output + wmsa(x)
            print(f'------------ output: {output.shape}')
            return output / len(self.pwmsa)

        elif self.fuse_mehod == 'split_attn':
            window_list = []
            for wmsa in self.pwmsa:
                window_list.append(wmsa(x))
            
            for i, window in enumerate(window_list):
                print(f'------------ window {i}: {window.shape}')
            return self.split_attn(window_list)
        
        
        
if __name__ == '__main__':
    # test BaseWindowAttention
    B, L, H, W, C = 2, 3, 48, 176, 256
    x = torch.randn(B, L, H, W, C)
    
    # embed_dim = C
    # num_heads = 4
    # dim_per_head = 64
    # drop_out = 0.1
    # window_size = 16
    # relative_pos_embedding = True
    # bwa = BaseWindowAttention(embed_dim, 
    #                         num_heads, 
    #                         dim_per_head, 
    #                         drop_out, 
    #                         window_size, 
    #                         relative_pos_embedding)
    
    pwa = PyramidWindowAttention(C, [16, 8, 4], [16, 32, 64], 0.3, [4, 8, 16], True, 'split_attn')
                                # dim, heads, dim_heads, drop_out, window_size, relative_pos_embedding, fuse_method
    
    out = pwa(x)
    print(out.shape)