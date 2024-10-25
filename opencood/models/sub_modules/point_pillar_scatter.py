import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']  # nx: 704, ny: 192, nz: 1 but where is this coming from????

        assert self.nz == 1

    def forward(self, batch_dict):
        '''
        Get pillar_features [n, c] 
        and 
        Transform to spatial_features [N, C, H, W]
        -- This is not really 'transforming' voxel to spatial. 
        -- It's more like getting the voxel coordinates associated with the pillar features -> then 'scattering' them to the spatial grid.
        '''
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords'] # [n, 64], [n, 4]
        '''
        Coords: [n, 4], each row contains:
            - The batch index.
            - The x, y, and z coordinates in the voxel grid.
        '''
        #print(f'pillar_features.shape: {pillar_features.shape}')
        #print(f'coords.shape: {coords.shape}')
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        #print(f'batch_size: {batch_size}')

        for batch_idx in range(batch_size):
            #print(f'------------------------batch_idx: {batch_idx}------------------------')
            #print(f'nx: {self.nx}, ny: {self.ny}, nz: {self.nz}')
            spatial_feature = torch.zeros(self.num_bev_features,
                                          self.nz * self.nx * self.ny,
                                            dtype=pillar_features.dtype,
                                            device=pillar_features.device)
            #print(f'spatial_feature.shape: {spatial_feature.shape}')

            batch_mask = coords[:, 0] == batch_idx
            #print(f'batch_mask.shape: {batch_mask.shape}')
            
            this_coords = coords[batch_mask, :]
            #print(f'this_coords.shape: {this_coords.shape}')

            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            #print(f'indices.shape: {indices.shape}')
            
            indices = indices.type(torch.long)
            #print(f'indices.shape: {indices.shape}')
            
            pillars = pillar_features[batch_mask, :]
            #print(f'pillars.shape: {pillars.shape}')
            
            pillars = pillars.t()
            #print(f'pillars.shape: {pillars.shape}')
            
            spatial_feature[:, indices] = pillars
            #print(f'spatial_feature.shape: {spatial_feature.shape}')
            
            batch_spatial_features.append(spatial_feature)
            #print(f'batch_spatial_features: {len(batch_spatial_features)}')

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, 
                                                                self.num_bev_features * self.nz, 
                                                                self.ny, 
                                                                self.nx)
        #print(f'batch_spatial_features.shape: {batch_spatial_features.shape}')
        
        batch_dict['spatial_features'] = batch_spatial_features

        return batch_dict

