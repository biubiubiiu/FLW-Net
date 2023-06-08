import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from pytorch_msssim import SSIM


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, x, y):
        return 1 - self.ssim(x, y)


def _cosine_distance_and_angle(x, y, dim, eps=1e-5):
    sim = F.cosine_similarity(x, y, dim=dim, eps=eps)
    distance = 1 - sim.mean()
    angle = torch.acos(sim.clip(min=-1 + eps, max=1 - eps)).mean()
    return distance + angle


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x, y):
        return _cosine_distance_and_angle(x, y, dim=1)


class BrightnessLoss(nn.Module):
    def __init__(self, n_patch_horizontal, n_patch_vertical):
        super(BrightnessLoss, self).__init__()
        self.n_patch_horizontal = n_patch_horizontal
        self.n_patch_vertical = n_patch_vertical

    def forward(self, x, y):
        tiles_x = rearrange(
            x,
            'b c (n1 h) (n2 w) -> (b n1 n2) c h w',
            n1=self.n_patch_vertical,
            n2=self.n_patch_horizontal,
        )
        tiles_y = rearrange(
            y,
            'b c (n1 h) (n2 w) -> (b n1 n2) c h w',
            n1=self.n_patch_vertical,
            n2=self.n_patch_horizontal,
        )

        min_x = reduce(tiles_x, 'b c h w -> b c', reduction='min').abs().detach()
        min_y = reduce(tiles_y, 'b c h w -> b c', reduction='min').abs().detach()

        shifted_x = tiles_x - min_x[..., None, None]
        shifted_y = tiles_y - min_y[..., None, None]

        return _cosine_distance_and_angle(shifted_x.flatten(2), shifted_y.flatten(2), dim=-1) + \
            _cosine_distance_and_angle(shifted_x.flatten(1), shifted_y.flatten(1), dim=-1)


def _get_spatial_gradient_kernel():
    kernel_x = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]).float()
    kernel_y = kernel_x.transpose(0, 1)
    kernel = torch.stack([kernel_x, kernel_y])
    return kernel[:, None, ...]  # OIHW


class GradLoss(nn.Module):
    def __init__(self, n_patch_horizontal, n_patch_vertical):
        super(GradLoss, self).__init__()
        self.n_patch_horizontal = n_patch_horizontal
        self.n_patch_vertical = n_patch_vertical

        kernel = _get_spatial_gradient_kernel()
        self.register_buffer('kernel', kernel)

    def get_gradients(self, x):
        tmp_kernel = self.kernel.repeat(x.size(1), 1, 1, 1).to(x)
        grad = F.conv2d(x, tmp_kernel, groups=x.size(1), padding=1)
        return grad

    def forward(self, x, y):
        grad_x = torch.abs(self.get_gradients(x))
        grad_y = torch.abs(self.get_gradients(y))

        tiles_grad_x = rearrange(
            grad_x,
            'b c (n1 h) (n2 w) -> (b n1 n2) c h w',
            n1=self.n_patch_vertical,
            n2=self.n_patch_horizontal,
        )
        tiles_grad_y = rearrange(
            grad_y,
            'b c (n1 h) (n2 w) -> (b n1 n2) c h w',
            n1=self.n_patch_vertical,
            n2=self.n_patch_horizontal,
        )

        min_grad_x = reduce(tiles_grad_x, 'b c h w -> b c', reduction='min').abs().detach()
        min_grad_y = reduce(tiles_grad_y, 'b c h w -> b c', reduction='min').abs().detach()

        shifted_grad_x = tiles_grad_x - min_grad_x[..., None, None]
        shifted_grad_y = tiles_grad_y - min_grad_y[..., None, None]

        return _cosine_distance_and_angle(shifted_grad_x.flatten(2), shifted_grad_y.flatten(2), dim=-1) + \
            _cosine_distance_and_angle(shifted_grad_x.flatten(1), shifted_grad_y.flatten(1), dim=-1)
