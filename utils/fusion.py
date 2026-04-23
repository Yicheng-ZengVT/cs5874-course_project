import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_layer_anomaly_maps(fs_list, ft_list, out_size):
    layer_maps = []
    for fs, ft in zip(fs_list, ft_list):
        a_map = 1.0 - F.cosine_similarity(fs, ft, dim=1)
        a_map = a_map.unsqueeze(1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        layer_maps.append(a_map)
    return layer_maps


def fuse_maps_baseline(layer_maps, mode='sum'):
    stack = torch.cat(layer_maps, dim=1)
    if mode == 'sum':
        return stack.sum(dim=1, keepdim=True)
    if mode == 'mean':
        return stack.mean(dim=1, keepdim=True)
    if mode == 'max':
        return stack.max(dim=1, keepdim=True).values
    if mode == 'mul':
        return stack.prod(dim=1, keepdim=True)
    raise ValueError(f"Unknown fusion mode: {mode}")


class FixedScalarFusion(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_maps, return_weights=False):
        weights = torch.softmax(self.logits, dim=0)
        fused = 0.0
        for i, amap in enumerate(layer_maps):
            fused = fused + weights[i] * amap
        if return_weights:
            return fused, weights.view(1, -1)
        return fused


class GlobalConditionedFusion(nn.Module):
    def __init__(self, channels=(256, 512, 1024), hidden_dim=256):
        super().__init__()
        gate_in_dim = sum([2 * c + 1 for c in channels])
        self.mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(channels))
        )

    def forward(self, fs_list, ft_list, layer_maps, return_weights=False):
        pooled = []
        for fs, ft, amap in zip(fs_list, ft_list, layer_maps):
            pooled.append(F.adaptive_avg_pool2d(fs, 1).flatten(1))
            pooled.append(F.adaptive_avg_pool2d(ft, 1).flatten(1))
            pooled.append(F.adaptive_avg_pool2d(amap, 1).flatten(1))
        gate_in = torch.cat(pooled, dim=1)
        weights = torch.softmax(self.mlp(gate_in), dim=1)

        fused = 0.0
        for i, amap in enumerate(layer_maps):
            fused = fused + weights[:, i].view(-1, 1, 1, 1) * amap
        if return_weights:
            return fused, weights
        return fused


def weight_entropy_regularizer(weights):
    eps = 1e-8
    entropy = -(weights * (weights + eps).log()).sum(dim=1).mean()
    return -entropy
