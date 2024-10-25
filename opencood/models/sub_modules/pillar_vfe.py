"""
Pillar VFE, credits to OpenPCDet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']

        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters']
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):
        '''
        Get voxel_feature [B, 32, 4] and
        Transform to pillar_feature [B, 64]
        '''

        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']  # (B, 32, 4), (B), (B, 4)

        # print(f'voxel_features.shape: {voxel_features.shape}')
        # print(f'voxel_num_points.shape: {voxel_num_points.shape}')
        # print(f'coords.shape: {coords.shape}')
        
        
        ### Normalization within each voxel
        '''
        points_mean: Computes the mean position (centroid) for the first three features (x, y, z coordinates) of each voxel. This results in a [B, 1, 3] tensor.
        f_cluster: Captures the deviation of each point’s position from the mean. This gives relative point positions.
        Goal:   Normalization within the voxel. 
                Compute the relative position of each point wrt the mean position of all points WITHIN THE VOXEL.
                Centering around each voxel's local mean.
                Captures the distribution of points within a voxel.
        '''
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean  # [B, 32, 3]


        ### Normalization wrt global voxel grid.
        '''
        Goal:   Centering relative to the voxel grid's spatial center.
                Centers points based on the voxel’s location in the GLOBAL COORDINATE SYSTEM.
                Make the network focus on geometry within each voxel, ignoring global translation.
        '''        
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        
        '''
            f_cluster captures local point spread (how points are arranged relative to their centroid inside a voxel), 
            f_center focuses on local geometry relative to the voxel’s center in the grid. 
            Both normalizations help the model focus on local features rather than being affected by absolute coordinates.
        '''
        # print(f'f_cluster.shape: {f_cluster.shape}')
        # print(f'f_center.shape: {f_center.shape}')
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2,
                                     keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)  # [B, 32, 10]


        '''
            get_paddings_indicator() creates a mask to identify valid points versus padding in each voxel. 
            This ensures the network only processes real data and ignores padded values. 
            The mask aligns with the number of points per voxel and ensures consistent shapes across the batch.
        '''
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask    # [B, 32, 10]   # 10 = (x,y,z,intensity) , f_cluster, f_center = 4 + 3 + 3
        
        for pfn in self.pfn_layers: # iterates once
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features    # [B, 64]
        return batch_dict   # [B, 64]
