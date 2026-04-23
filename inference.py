import os
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch

from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_test import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer
from utils.fusion import FixedScalarFusion, GlobalConditionedFusion
from dataset.dataset import MVTecDataset_test, get_data_transforms


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--checkpoint_folder', required=True, type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--classes', nargs='+', default=['carpet', 'leather'])
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--save_gate_details', action='store_true')
    return parser.parse_args()


def build_fusion_from_checkpoint(ckp, device):
    fusion_mode = ckp.get('fusion_mode', 'sum')
    fusion_module = None
    if fusion_mode == 'fixed_scalar':
        fusion_module = FixedScalarFusion(num_layers=3).to(device)
        fusion_module.load_state_dict(ckp['fusion'])
    elif fusion_mode == 'global_gate':
        fusion_module = GlobalConditionedFusion(channels=(256, 512, 1024), hidden_dim=256).to(device)
        fusion_module.load_state_dict(ckp['fusion'])
    return fusion_mode, fusion_module


def inference(_class_, pars):
    os.makedirs(pars.checkpoint_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)
    test_path = os.path.join(pars.data_root, _class_)
    if not os.path.isdir(test_path):
        raise FileNotFoundError(f'test_path not found: {test_path}')

    checkpoint_class = os.path.join(pars.checkpoint_folder, _class_, f'wres50_{_class_}.pth')
    if not os.path.isfile(checkpoint_class):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_class}')

    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False,
        num_workers=pars.num_workers, pin_memory=True, drop_last=False
    )

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    proj_layer = MultiProjectionLayer(base=64).to(device)

    ckp = torch.load(checkpoint_class, map_location='cpu')
    proj_layer.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])

    fusion_mode, fusion_module = build_fusion_from_checkpoint(ckp, device)

    if pars.save_gate_details and fusion_module is not None:
        auroc_px, auroc_sp, aupro_px, details_df = evaluation_multi_proj(
            encoder, proj_layer, bn, decoder, test_dataloader, device,
            fusion_mode=fusion_mode, fusion_module=fusion_module, return_details=True
        )
        details_df.to_csv(os.path.join(pars.checkpoint_folder, _class_, 'inference_gate_details.csv'), index=False)
    else:
        auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(
            encoder, proj_layer, bn, decoder, test_dataloader, device,
            fusion_mode=fusion_mode, fusion_module=fusion_module
        )

    print(f'{_class_}: fusion={fusion_mode}, Sample Auroc: {auroc_sp:.4f}, Pixel Auroc:{auroc_px:.4f}, Pixel Aupro: {aupro_px:.4f}')
    return auroc_sp, auroc_px, aupro_px


if __name__ == '__main__':
    pars = get_args()
    setup_seed(pars.seed)

    metrics = {'class': [], 'AUROC_sample': [], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px = inference(c, pars)
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        pd.DataFrame(metrics).to_csv(os.path.join(pars.checkpoint_folder, 'metrics_checkpoints.csv'), index=False)
