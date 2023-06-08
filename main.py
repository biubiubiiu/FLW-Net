import os.path as osp
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import BrightnessLoss, ColorLoss, GradLoss, SSIMLoss
from model import FLWNet
from utils import AverageMeter, LOLDataset, parse_args


def evaluate_model(model, data_loader, save_dir, exp_mean=None, num_epoch=None):
    model.eval()
    avg_psnr, avg_ssim = AverageMeter('PSNR'), AverageMeter('SSIM')

    with torch.inference_mode():
        test_bar = tqdm(data_loader, dynamic_ncols=True)
        for data in test_bar:
            lq, gt = data['input'].to(device), data['target'].to(device)
            mean = exp_mean or torch.mean(torch.max(gt, dim=1)[0], dim=[-1, -2])
            out = model(lq, mean).squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()

            out = (out * 255).astype(np.uint8)
            gt = (gt * 255).astype(np.uint8)

            current_psnr = compute_psnr(out, gt)
            current_ssim = compute_ssim(
                out, gt, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False
            )

            avg_psnr.update(current_psnr)
            avg_ssim.update(current_ssim)

            save_path = Path(save_dir, 'val_result', str(num_epoch), data['fn'][0])
            save_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(out).save(save_path)
            test_bar.set_description(
                f'Test Epoch: [{num_epoch}] ' f'PSNR: {avg_psnr.avg:.2f} SSIM: {avg_ssim.avg:.4f}'
            )

    return avg_psnr.avg, avg_ssim.avg


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cpu') if args.cpu else torch.device('cuda')

    test_dataset = LOLDataset(osp.join(args.data_path, 'eval15'), training=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    save_path = Path(args.save_path)

    model = FLWNet().to(device)
    if args.phase == 'test':
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        exp_mean = None if args.use_gt_mean else args.exp_mean
        evaluate_model(model, test_loader, save_path, exp_mean=exp_mean, num_epoch='final')
    else:
        losses = {
            'l1_loss': [nn.L1Loss(), 1.0],
            'ssim_loss': [SSIMLoss(), 1.0],
            'color_loss': [ColorLoss(), 1.0],
            'brightness_loss': [BrightnessLoss(1, 1), 1.0],
            'grad_loss': [GradLoss(1, 1), 1.0],
        }
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_dataset = LOLDataset(
            osp.join(args.data_path, 'our485'), patch_size=args.patch_size, training=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )

        train_bar = tqdm(range(1, args.num_epoch + 1), dynamic_ncols=True)
        for n_epoch in train_bar:
            model.train()
            epoch_loss = 0
            for data in train_loader:  # train
                lq, gt = data['input'].to(device), data['target'].to(device)
                gt_mean = torch.mean(torch.max(gt, dim=1)[0], dim=[-1, -2])
                out = model(lq, gt_mean)
                loss = sum([(weight * loss(out, gt)) for loss, weight in losses.values()])
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_bar.set_description(f'Train Epoch: [{n_epoch}/{args.num_epoch+1}] Loss: {epoch_loss:.3f}')

            if n_epoch % args.eval_step == 0:  # evaluate
                val_psnr, val_ssim = evaluate_model(
                    model, test_loader, save_path, exp_mean=None, num_epoch=n_epoch
                )
                with save_path.joinpath('record.txt').open(mode='a+') as f:
                    f.write(f'Epoch: {n_epoch} PSNR:{val_psnr:.2f} SSIM:{val_ssim:.4f}\n')

            if n_epoch % args.save_step == 0:
                torch.save(model.state_dict(), save_path.joinpath('checkpoints', f'{n_epoch}.pth'))
