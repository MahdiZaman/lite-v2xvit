# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import numpy as np

from einops import rearrange
from opencood.utils.common_utils import torch_tensor_to_numpy


def regroup(dense_feature, record_len, max_len):
    """
    Regroup the data based on the record_len.

    Parameters
    ----------
    dense_feature : torch.Tensor
        N, C, H, W
    record_len : list
        [sample1_len, sample2_len, ...]
    max_len : int
        Maximum cav number

    Returns
    -------
    regroup_feature : torch.Tensor
        B, L, C, H, W
    """
    cum_sum_len = list(np.cumsum(torch_tensor_to_numpy(record_len)))
    # with B = 2, L = 3, the above line will return [3, 5]
    
    split_features = torch.tensor_split(dense_feature,
                                        cum_sum_len[:-1])
    # split_features is a tuple of tensors, each tensor is a slice of dense_feature
    
    regroup_features = []
    mask = []

    # print(f'split_features length: {len(split_features)}')
    # split_features is made by slicing dense_feature along Batch axis, so it has length B 
    for split_feature in split_features:    # for each sample in a batch
        ## M = number of cavs in current sample (max val of M is set by config yaml)
        # M, C, H, W
        # print(f'split_feature.shape before padding: {split_feature.shape}')
        
        feature_shape = split_feature.shape

        # the maximum M is 5 as most 5 cavs
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)
        # print(f'len(mask): {len(mask)}')
        # print(f'mask[-1]: {len(mask[-1])}')

        padding_tensor = torch.zeros(padding_len, feature_shape[1],
                                     feature_shape[2], feature_shape[3])
        padding_tensor = padding_tensor.to(split_feature.device)

        split_feature = torch.cat([split_feature, padding_tensor],
                                  dim=0)
        # print(f'split_feature.shape after padding: {split_feature.shape}')
        # 1, 5C, H, W
        split_feature = split_feature.view(-1,
                                           feature_shape[2],
                                           feature_shape[3]).unsqueeze(0)
        # print(f'split_feature.shape reshaped: {split_feature.shape}')
        regroup_features.append(split_feature)
        # print(f'len(regroup_features): {len(regroup_features)}')
        # print(f'regroup_features[-1].shape: {regroup_features[-1].shape}')
        
    # B, 5C, H, W
    regroup_features = torch.cat(regroup_features, dim=0)
    # B, L, C, H, W
    regroup_features = rearrange(regroup_features,
                                 'b (l c) h w -> b l c h w',
                                 l=max_len)
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)
    # print(f'mask type: {type(mask)} -- mask.shape: {mask.shape}')
    # print(f'unique mask values: {torch.unique(mask)}')
    # print(f'mask: {mask}')
    
    # print(f'---------- Final regroup_features.shape: {regroup_features.shape}')

    return regroup_features, mask
