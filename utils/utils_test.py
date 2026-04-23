import torch
import cv2
import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.ndimage import gaussian_filter
from skimage import measure
from sklearn.metrics import roc_auc_score, auc
from statistics import mean
import warnings

from utils.fusion import compute_layer_anomaly_maps, fuse_maps_baseline

warnings.filterwarnings('ignore')


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='sum'):
    layer_maps = compute_layer_anomaly_maps(fs_list, ft_list, out_size)
    fused = fuse_maps_baseline(layer_maps, mode=amap_mode)
    anomaly_map = fused[0, 0].detach().cpu().numpy()
    a_map_list = [m[0, 0].detach().cpu().numpy() for m in layer_maps]
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    if a_max - a_min < 1e-12:
        return np.zeros_like(image)
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    return cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)


def evaluation_multi_proj(
    encoder, proj, bn, decoder, dataloader, device,
    fusion_mode='sum', fusion_module=None, sigma=4, return_details=False
):
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    if fusion_module is not None:
        fusion_module.eval()

    gt_list_px, pr_list_px = [], []
    gt_list_sp, pr_list_sp = [], []
    aupro_list = []
    detail_rows = []

    with torch.no_grad():
        for (img, gt, label, img_type, file_name) in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            layer_maps = compute_layer_anomaly_maps(inputs, outputs, img.shape[-1])

            if fusion_module is None:
                fused = fuse_maps_baseline(layer_maps, mode=fusion_mode)
                gate_weights = None
            else:
                if fusion_mode == 'fixed_scalar':
                    fused, gate_weights = fusion_module(layer_maps, return_weights=True)
                elif fusion_mode == 'global_gate':
                    fused, gate_weights = fusion_module(inputs, outputs, layer_maps, return_weights=True)
                else:
                    raise ValueError(f'Unsupported learned fusion_mode: {fusion_mode}')

            anomaly_map = fused[0, 0].detach().cpu().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=sigma)

            gt = gt.clone()
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            gt_np = gt.cpu().numpy().astype(int)

            if label.item() != 0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int), anomaly_map[np.newaxis, :, :]))

            gt_list_px.extend(gt_np.ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt_np))
            pr_list_sp.append(np.max(anomaly_map))

            if return_details:
                row = {
                    'file_name': file_name[0] if isinstance(file_name, (list, tuple)) else str(file_name),
                    'img_type': img_type[0] if isinstance(img_type, (list, tuple)) else str(img_type),
                    'label': int(label.item()),
                    'image_score': float(np.max(anomaly_map)),
                }
                for i, lm in enumerate(layer_maps):
                    row[f'layer_{i+1}_score'] = float(lm.max().item())
                if gate_weights is not None:
                    gw = gate_weights[0].detach().cpu().numpy()
                    for i in range(len(gw)):
                        row[f'gate_w_{i+1}'] = float(gw[i])
                detail_rows.append(row)

    auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    aupro_px = round(float(np.mean(aupro_list)), 4) if len(aupro_list) > 0 else 0.0

    if return_details:
        return auroc_px, auroc_sp, aupro_px, pd.DataFrame(detail_rows)
    return auroc_px, auroc_sp, aupro_px


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:
    assert isinstance(amaps, ndarray)
    assert isinstance(masks, ndarray)
    assert amaps.ndim == 3
    assert masks.ndim == 3
    assert amaps.shape == masks.shape

    d = {'pro': [], 'fpr': [], 'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    if max_th - min_th < 1e-12:
        return 0.0
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / max(region.area, 1))

        inverse_masks = 1 - masks
        denom = inverse_masks.sum()
        fpr = 0.0 if denom == 0 else np.logical_and(inverse_masks, binary_amaps).sum() / denom

        d['pro'].append(mean(pros) if len(pros) > 0 else 0.0)
        d['fpr'].append(fpr)
        d['threshold'].append(th)

    df = pd.DataFrame(d)
    df = df[df['fpr'] < 0.3].copy()
    if len(df) < 2 or df['fpr'].max() <= 0:
        return 0.0
    df['fpr'] = df['fpr'] / df['fpr'].max()
    return float(auc(df['fpr'], df['pro']))
