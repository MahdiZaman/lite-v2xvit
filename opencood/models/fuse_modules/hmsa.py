import torch
from torch import nn

from einops import rearrange


class HGTCavAttention(nn.Module):
    def __init__(self, dim, heads, num_types=2,
                 num_relations=4, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_types = num_types

        self.attend = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(dropout)
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for t in range(num_types):
            self.k_linears.append(nn.Linear(dim, inner_dim))
            self.q_linears.append(nn.Linear(dim, inner_dim))
            self.v_linears.append(nn.Linear(dim, inner_dim))
            self.a_linears.append(nn.Linear(inner_dim, dim))

        self.relation_att = nn.Parameter(
            torch.Tensor(num_relations, heads, dim_head, dim_head))
        self.relation_msg = nn.Parameter(
            torch.Tensor(num_relations, heads, dim_head, dim_head))

        torch.nn.init.xavier_uniform(self.relation_att)
        torch.nn.init.xavier_uniform(self.relation_msg)

    def to_qkv(self, x, types):
        # x: (B,H,W,L,C)
        # types: (B,L)
        # print(f'types: {types}')
        q_batch = []
        k_batch = []
        v_batch = []

        for b in range(x.shape[0]):
            q_list = []
            k_list = []
            v_list = []

            for i in range(x.shape[-2]):
                # (H,W,1,C)
                q_list.append(
                    self.q_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
                k_list.append(
                    self.k_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
                v_list.append(
                    self.v_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
            # (1,H,W,L,C)
            q_batch.append(torch.cat(q_list, dim=2).unsqueeze(0))
            k_batch.append(torch.cat(k_list, dim=2).unsqueeze(0))
            v_batch.append(torch.cat(v_list, dim=2).unsqueeze(0))
        # (B,H,W,L,C)
        q = torch.cat(q_batch, dim=0)
        k = torch.cat(k_batch, dim=0)
        v = torch.cat(v_batch, dim=0)
        return q, k, v

    def get_relation_type_index(self, type1, type2):
        return type1 * self.num_types + type2

    def get_hetero_edge_weights(self, x, types):
        w_att_batch = []
        w_msg_batch = []

        for b in range(x.shape[0]):
            w_att_list = []
            w_msg_list = []

            for i in range(x.shape[-2]):
                w_att_i_list = []
                w_msg_i_list = []

                for j in range(x.shape[-2]):
                    e_type = self.get_relation_type_index(types[b, i],
                                                          types[b, j])
                    w_att_i_list.append(self.relation_att[e_type].unsqueeze(0))
                    w_msg_i_list.append(self.relation_msg[e_type].unsqueeze(0))
                w_att_list.append(torch.cat(w_att_i_list, dim=0).unsqueeze(0))
                w_msg_list.append(torch.cat(w_msg_i_list, dim=0).unsqueeze(0))

            w_att_batch.append(torch.cat(w_att_list, dim=0).unsqueeze(0))
            w_msg_batch.append(torch.cat(w_msg_list, dim=0).unsqueeze(0))

        # (B,M,L,L,C_head,C_head)
        w_att = torch.cat(w_att_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        w_msg = torch.cat(w_msg_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        return w_att, w_msg

    def to_out(self, x, types):
        out_batch = []
        for b in range(x.shape[0]):
            out_list = []
            for i in range(x.shape[-2]):
                out_list.append(
                    self.a_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
            out_batch.append(torch.cat(out_list, dim=2).unsqueeze(0))
        out = torch.cat(out_batch, dim=0)
        return out

    def forward(self, x, mask, prior_encoding):
        # print(f'HMSA start. x: {x.shape}, mask: {mask.shape}, prior_encoding: {prior_encoding.shape}')
        
        x = x.permute(0, 2, 3, 1, 4)    # x: (B, L, H, W, C) -> (B, H, W, L, C)
        
        mask = mask.unsqueeze(1)    # mask: (B, 1, H, W, 1, L)
        
        velocities, dts, types = [itm.squeeze(-1) for itm in
                                prior_encoding[:, :, 0, 0, :].split(
                                    [1, 1, 1], dim=-1)]     # prior_encoding: (B,L,H,W,3)
        
        # print(f'Attention computation start. x: {x.shape}, mask: {mask.shape}, prior_encoding: {prior_encoding.shape}')
        types = types.to(torch.int) # (B,L)
        dts = dts.to(torch.int)
        
        # features transformed into tokens based on their type
        # # print(f'----to_qkv----')
        qkv = self.to_qkv(x, types) # qkv: torch.Size([2, 48, 176, 2, 256]), torch.Size([2, 48, 176, 2, 256]), torch.Size([2, 48, 176, 2, 256])
        # # print(f'qkv: {qkv[0].shape}, {qkv[1].shape}, {qkv[2].shape}')
        
        
        # (B,M,L,L,C_head,C_head)
        w_att, w_msg = self.get_hetero_edge_weights(x, types)
        # print(f'--- w_att: {w_att.shape}, w_msg: {w_msg.shape}')
        
        
        # q: (B, M, H, W, L, C)
        # print(f'qkv: {qkv[0].shape}, {qkv[1].shape}, {qkv[2].shape}')
        q, k, v = map(lambda t: rearrange(t, 'b h w l (m c) -> b m h w l c', m=self.heads), (qkv))
        # print(f'--- q: {q.shape}, k: {k.shape}, v: {v.shape}')
        
        # attention, (B, M, H, W, L, L)
        # print(f'q: {q.shape}, w_att: {w_att.shape}, k: {k.shape}')
        att_map = torch.einsum(
            'b m h w i p, b m i j p q, bm h w j q -> b m h w i j',
            [q, w_att, k]) * self.scale
        # add mask
        att_map = att_map.masked_fill(mask == 0, -float('inf'))
        # softmax
        att_map = self.attend(att_map)
        # print(f'--- att_map: {att_map.shape}')
        

        # out:(B, M, H, W, L, C_head)
        # print(f'v: {v.shape}, w_msg: {w_msg.shape}')
        v_msg = torch.einsum('b m i j p c, b m h w j p -> b m h w i j c',
                            w_msg, v)
        # print(f'--- v_msg: {v_msg.shape}')
        
        out = torch.einsum('b m h w i j, b m h w i j c -> b m h w i c',
                        att_map, v_msg)
        # print(f'--- out: {out.shape}')
        

        out = rearrange(out, 'b m h w l c -> b h w l (m c)',
                        m=self.heads)
        # print(f'--- out after rearrange: {out.shape}')
        out = self.to_out(out, types)
        out = self.drop_out(out)
        # (B L H W C)
        out = out.permute(0, 3, 1, 2, 4)
        # print(f'out: {out.shape}')
        return out
    
    
    
if __name__ == '__main__':
    dim = 256
    # use_hetero = true
    # use_RTE = true
    RTE_ratio = 2
    heads = 8
    dim_head = 32
    dropout = 0.3
    
    x = torch.rand([2, 2, 48, 176, 256]) 
    mask = torch.rand([2, 48, 176, 1, 2])
    prior_encoding = torch.rand([2, 2, 48, 176, 3])

    att = HGTCavAttention(dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout)
    
    out = att(x, mask, prior_encoding)
    print('out: ', out.shape)