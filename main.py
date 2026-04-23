import os
import json
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_test import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion
from utils.fusion import compute_layer_anomaly_maps, FixedScalarFusion, GlobalConditionedFusion, weight_entropy_regularizer
from dataset.dataset import MVTecDataset_test, MVTecDataset_train, get_data_transforms


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_epochs(_class_):
    epoch_map = {
        'carpet': 10, 'leather': 10, 'grid': 260, 'tile': 260, 'wood': 100,
        'cable': 240, 'capsule': 300, 'hazelnut': 160, 'metal_nut': 160,
        'screw': 280, 'toothbrush': 280, 'transistor': 300, 'zipper': 300,
        'pill': 200, 'bottle': 200
    }
    return epoch_map.get(_class_, 200)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--save_folder', default='./RDpp_checkpoint_result', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--proj_lr', default=1e-3, type=float)
    parser.add_argument('--distill_lr', default=5e-3, type=float)
    parser.add_argument('--fusion_lr', default=1e-3, type=float)
    parser.add_argument('--weight_proj', default=0.2, type=float)
    parser.add_argument('--fusion_pseudo_weight', default=0.2, type=float)
    parser.add_argument('--fusion_entropy_weight', default=1e-3, type=float)
    parser.add_argument('--fusion_warmup_epochs', default=5, type=int)
    parser.add_argument('--classes', nargs='+', default=['carpet', 'leather'])
    parser.add_argument('--fusion_mode', default='sum', choices=['sum', 'mean', 'max', 'mul', 'fixed_scalar', 'global_gate'])
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--save_gate_details', action='store_true')
    return parser.parse_args()


def build_fusion_module(pars, device):
    fusion_module = None
    optimizer_fusion = None
    if pars.fusion_mode == 'fixed_scalar':
        fusion_module = FixedScalarFusion(num_layers=3).to(device)
    elif pars.fusion_mode == 'global_gate':
        fusion_module = GlobalConditionedFusion(channels=(256, 512, 1024), hidden_dim=256).to(device)
    if fusion_module is not None:
        optimizer_fusion = torch.optim.Adam(fusion_module.parameters(), lr=pars.fusion_lr, betas=(0.5, 0.999))
    return fusion_module, optimizer_fusion


def train(_class_, pars):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)
    train_path = os.path.join(pars.data_root, _class_, 'train')
    test_path = os.path.join(pars.data_root, _class_)

    if not os.path.isdir(train_path):
        raise FileNotFoundError(f'train_path not found: {train_path}')
    if not os.path.isdir(test_path):
        raise FileNotFoundError(f'test_path not found: {test_path}')

    class_save_dir = os.path.join(pars.save_folder, _class_)
    os.makedirs(class_save_dir, exist_ok=True)
    save_model_path = os.path.join(class_save_dir, f'wres50_{_class_}.pth')

    train_data = MVTecDataset_train(root=train_path, transform=data_transform)
    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=pars.batch_size, shuffle=True,
        num_workers=pars.num_workers, pin_memory=True, drop_last=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False,
        num_workers=pars.num_workers, pin_memory=True, drop_last=False
    )

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    proj_layer = MultiProjectionLayer(base=64).to(device)

    proj_loss = Revisit_RDLoss()
    optimizer_proj = torch.optim.Adam(proj_layer.parameters(), lr=pars.proj_lr, betas=(0.5, 0.999))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=pars.distill_lr, betas=(0.5, 0.999))
    fusion_module, optimizer_fusion = build_fusion_module(pars, device)

    best_score = -1.0
    best_epoch = 0
    best_auroc_px = 0.0
    best_auroc_sp = 0.0
    best_aupro_px = 0.0

    auroc_px_list, auroc_sp_list, aupro_px_list = [], [], []
    loss_proj_list, loss_distill_list, total_loss_list, loss_fusion_list = [], [], [], []

    num_epoch = get_default_epochs(_class_)
    print(f'with class {_class_}, Training with {num_epoch} Epoch')
    accumulation_steps = 2

    for epoch in tqdm(range(1, num_epoch + 1)):
        bn.train()
        proj_layer.train()
        decoder.train()
        if fusion_module is not None:
            fusion_module.train()

        loss_proj_running = 0.0
        loss_distill_running = 0.0
        loss_fusion_running = 0.0
        total_loss_running = 0.0

        optimizer_proj.zero_grad(set_to_none=True)
        optimizer_distill.zero_grad(set_to_none=True)
        if optimizer_fusion is not None:
            optimizer_fusion.zero_grad(set_to_none=True)

        for i, (img, img_noise, _) in enumerate(train_dataloader):
            img = img.to(device, non_blocking=True)
            img_noise = img_noise.to(device, non_blocking=True)

            with torch.no_grad():
                inputs = encoder(img)
                inputs_noise = encoder(img_noise)

            feature_space_noise, feature_space = proj_layer(inputs, features_noise=inputs_noise)
            outputs = decoder(bn(feature_space))

            L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)
            L_distill = loss_fucntion(inputs, outputs)
            loss = L_distill + pars.weight_proj * L_proj
            fusion_loss_value = 0.0

            if fusion_module is not None and epoch >= pars.fusion_warmup_epochs:
                noised_outputs = decoder(bn(feature_space_noise))
                clean_layer_maps = compute_layer_anomaly_maps(inputs, outputs, img.shape[-1])
                noisy_layer_maps = compute_layer_anomaly_maps(inputs_noise, noised_outputs, img.shape[-1])

                if pars.fusion_mode == 'fixed_scalar':
                    clean_fused, clean_weights = fusion_module(clean_layer_maps, return_weights=True)
                    noisy_fused, noisy_weights = fusion_module(noisy_layer_maps, return_weights=True)
                elif pars.fusion_mode == 'global_gate':
                    clean_fused, clean_weights = fusion_module(inputs, outputs, clean_layer_maps, return_weights=True)
                    noisy_fused, noisy_weights = fusion_module(inputs_noise, noised_outputs, noisy_layer_maps, return_weights=True)
                else:
                    raise ValueError(f'Unsupported fusion_mode {pars.fusion_mode}')

                reg = weight_entropy_regularizer(clean_weights) + weight_entropy_regularizer(noisy_weights)
                clean_score = clean_fused.mean()
                noisy_score = noisy_fused.mean()
                rank_loss = torch.relu(0.15 + clean_score - noisy_score)
                fusion_loss = pars.fusion_pseudo_weight * rank_loss + pars.fusion_entropy_weight * reg
                loss = loss + fusion_loss
                fusion_loss_value = float(fusion_loss.detach().cpu().item())

            loss = loss / accumulation_steps
            loss.backward()

            if ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_dataloader)):
                optimizer_proj.step()
                optimizer_distill.step()
                if optimizer_fusion is not None:
                    optimizer_fusion.step()
                optimizer_proj.zero_grad(set_to_none=True)
                optimizer_distill.zero_grad(set_to_none=True)
                if optimizer_fusion is not None:
                    optimizer_fusion.zero_grad(set_to_none=True)

            total_loss_running += float(loss.detach().cpu().item() * accumulation_steps)
            loss_proj_running += float(L_proj.detach().cpu().item())
            loss_distill_running += float(L_distill.detach().cpu().item())
            loss_fusion_running += fusion_loss_value

        if fusion_module is None:
            auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(
                encoder, proj_layer, bn, decoder, test_dataloader, device,
                fusion_mode=pars.fusion_mode, fusion_module=None
            )
            details_df = None
        else:
            if pars.save_gate_details:
                auroc_px, auroc_sp, aupro_px, details_df = evaluation_multi_proj(
                    encoder, proj_layer, bn, decoder, test_dataloader, device,
                    fusion_mode=pars.fusion_mode, fusion_module=fusion_module, return_details=True
                )
            else:
                auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(
                    encoder, proj_layer, bn, decoder, test_dataloader, device,
                    fusion_mode=pars.fusion_mode, fusion_module=fusion_module
                )
                details_df = None

        auroc_px_list.append(auroc_px)
        auroc_sp_list.append(auroc_sp)
        aupro_px_list.append(aupro_px)
        loss_proj_list.append(loss_proj_running)
        loss_distill_list.append(loss_distill_running)
        total_loss_list.append(total_loss_running)
        loss_fusion_list.append(loss_fusion_running)

        fig, ax = plt.subplots(4, 2, figsize=(10, 14))
        ax[0][0].plot(auroc_px_list); ax[0][0].set_title('auroc_px')
        ax[0][1].plot(auroc_sp_list); ax[0][1].set_title('auroc_sp')
        ax[1][0].plot(aupro_px_list); ax[1][0].set_title('aupro_px')
        ax[1][1].plot(loss_proj_list); ax[1][1].set_title('loss_proj')
        ax[2][0].plot(loss_distill_list); ax[2][0].set_title('loss_distill')
        ax[2][1].plot(total_loss_list); ax[2][1].set_title('total_loss')
        ax[3][0].plot(loss_fusion_list); ax[3][0].set_title('loss_fusion')
        ax[3][1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(class_save_dir, 'monitor_training.png'), dpi=100)
        plt.close(fig)

        print(f'Epoch {epoch}, Sample Auroc: {auroc_sp:.4f}, Pixel Auroc:{auroc_px:.4f}, Pixel Aupro: {aupro_px:.4f}')

        current_score = (auroc_px + auroc_sp + aupro_px) / 3.0
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_auroc_px = auroc_px
            best_auroc_sp = auroc_sp
            best_aupro_px = aupro_px

            save_dict = {
                'proj': proj_layer.state_dict(),
                'decoder': decoder.state_dict(),
                'bn': bn.state_dict(),
                'fusion_mode': pars.fusion_mode,
            }
            if fusion_module is not None:
                save_dict['fusion'] = fusion_module.state_dict()
            torch.save(save_dict, save_model_path)

            history_info = {
                'fusion_mode': pars.fusion_mode,
                'auroc_sp': best_auroc_sp,
                'auroc_px': best_auroc_px,
                'aupro_px': best_aupro_px,
                'epoch': best_epoch,
            }
            with open(os.path.join(class_save_dir, 'history.json'), 'w') as f:
                json.dump(history_info, f, indent=2)

            if details_df is not None:
                details_df.to_csv(os.path.join(class_save_dir, 'best_gate_details.csv'), index=False)

    return best_auroc_sp, best_auroc_px, best_aupro_px


if __name__ == '__main__':
    pars = get_args()
    print('Training with classes:', pars.classes)
    setup_seed(pars.seed)
    os.makedirs(pars.save_folder, exist_ok=True)

    metrics = {'class': [], 'AUROC_sample': [], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px = train(c, pars)
        print(f'Best score of class: {c}, Auroc sample: {auroc_sp:.4f}, Auroc pixel:{auroc_px:.4f}, Pixel Aupro: {aupro_px:.4f}')
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        pd.DataFrame(metrics).to_csv(os.path.join(pars.save_folder, 'metrics_results.csv'), index=False)
