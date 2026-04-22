import numpy as np


def fuse_anomaly_maps(a_map_list, fusion_mode='sum', weights=None):
    """
    Fuse a list of per layer anomaly maps into one final anomaly map.

    Args:
        a_map_list: list of numpy arrays, each shaped [H, W]
        fusion_mode: one of ['sum', 'mean', 'max', 'mul', 'a', 'weighted']
        weights: optional list/array of per-layer weights for 'weighted'

    Returns:
        anomaly_map: fused numpy array of shape [H, W]
    """
    if len(a_map_list) == 0:
        raise ValueError("a_map_list is empty")

    maps = np.stack(a_map_list, axis=0)  # [L, H, W]

    if fusion_mode in ['sum', 'a']:
        anomaly_map = np.sum(maps, axis=0)
    elif fusion_mode == 'mean':
        anomaly_map = np.mean(maps, axis=0)
    elif fusion_mode == 'max':
        anomaly_map = np.max(maps, axis=0)
    elif fusion_mode == 'mul':
        anomaly_map = np.prod(maps, axis=0)
    elif fusion_mode == 'weighted':
        if weights is None:
            raise ValueError("weights are required when fusion_mode='weighted'")
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array-like")
        if len(weights) != maps.shape[0]:
            raise ValueError(
                f"Expected {maps.shape[0]} weights, got {len(weights)}"
            )
        weights = weights / (weights.sum() + 1e-8)
        anomaly_map = np.sum(maps * weights[:, None, None], axis=0)
    else:
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    return anomaly_map