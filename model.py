import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce


def get_histogram(imgs, nbins):
    bs = imgs.size(0)
    hists = torch.zeros((bs, nbins + 2), requires_grad=False).to(imgs)
    for i in range(bs):
        min_val = reduce(imgs[i], 'c h w -> 1', reduction='min')
        max_val = reduce(imgs[i], 'c h w -> 1', reduction='max')
        counts = torch.histc(imgs[i], nbins, min=min_val.item(), max=max_val.item())

        hists[i, :nbins] = counts / counts.sum()
        hists[i, nbins:] = torch.cat([min_val, max_val])

    return hists


class HistBranch(nn.Module):
    def __init__(self, nbins, mid_channels, num_iters):
        super().__init__()

        self.nbins = nbins
        self.num_iters = num_iters

        self.g_conv1 = nn.Linear(self.nbins + 3, mid_channels)
        self.g_conv2 = nn.Linear(mid_channels, mid_channels)
        self.g_conv3 = nn.Linear(mid_channels + self.nbins + 3, mid_channels)
        self.g_conv4 = nn.Linear(mid_channels, mid_channels)
        self.g_conv5 = nn.Linear(mid_channels, self.num_iters)

    def forward(self, V_chanel, mu):
        hist = get_histogram(V_chanel, self.nbins)
        vec = torch.cat([hist, mu], dim=-1)

        hbf = F.leaky_relu(self.g_conv1(vec))
        hbf = F.leaky_relu(self.g_conv2(hbf))
        hbf = F.leaky_relu(self.g_conv3(torch.cat([hbf, vec], 1)))
        hbf = F.leaky_relu(self.g_conv4(hbf))
        alphas = F.leaky_relu(self.g_conv5(hbf))

        # S-curve mapping
        alpha_each_iter = torch.chunk(alphas, self.num_iters, dim=-1)
        out = V_chanel
        for alpha in alpha_each_iter:
            out = out + alpha[..., None, None] * (out - out**2)
        return out


class ImageBranch(nn.Module):
    def __init__(self, rescale_factor, mid_channels):
        super().__init__()

        self.rescale_factor = rescale_factor

        self.e_conv1 = nn.Conv2d(4, mid_channels, kernel_size=3, padding=1)
        self.e_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.e_conv3 = nn.Conv2d(mid_channels + 4, mid_channels, kernel_size=3, padding=1)
        self.e_conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2)
        self.e_conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2)
        self.e_conv6 = nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1)
        self.e_conv7 = nn.Conv2d(mid_channels, 3, kernel_size=3, padding=1)

    def forward(self, x, V_channel, enhanced_V):
        if self.rescale_factor > 1:
            h, w = V_channel.shape[-2:]  # input image shape
            ori_size = (h, w)
            new_size = (h // self.rescale_factor, w // self.rescale_factor)
            resized_V = F.interpolate(V_channel, size=new_size, mode='bilinear')
            resized_V = F.interpolate(resized_V, size=ori_size, mode='bilinear')
        else:
            resized_V = V_channel

        ibf = F.leaky_relu(self.e_conv1(torch.cat([x - resized_V / 2, resized_V / 2], dim=1)))
        ibf = F.leaky_relu(self.e_conv2(ibf))
        ibf = F.leaky_relu(self.e_conv3(torch.cat([ibf, x, enhanced_V], dim=1)))
        skip = ibf
        ibf = F.leaky_relu(self.e_conv4(ibf))
        ibf = F.leaky_relu(self.e_conv5(ibf))
        ibf = F.leaky_relu(self.e_conv6(torch.cat([skip, ibf], dim=1)))
        out = F.softplus(self.e_conv7(ibf))
        return out


class FLWNet(nn.Module):
    def __init__(self, rescale_factor=20, nbins=12, mid_channels=16, num_iters=7):
        super(FLWNet, self).__init__()

        self.hist_branch = HistBranch(nbins, mid_channels, num_iters)
        self.image_branch = ImageBranch(rescale_factor, mid_channels)

    def forward(self, x, expected_means):
        if not isinstance(expected_means, torch.Tensor):
            expected_means = torch.Tensor([expected_means]).to(x)

        expected_means = torch.reshape(expected_means, (x.shape[0], 1))
        V_channel, _ = torch.max(x, dim=1, keepdim=True)
        enhanced_V = self.hist_branch(V_channel, expected_means)
        out = self.image_branch(x, V_channel, enhanced_V)
        return out


if __name__ == '__main__':
    img = torch.randn(1, 3, 128, 128)
    net = FLWNet()
    print(net(img, 0.4).shape)

    from torchinfo import summary
    summary(net, input_size=(img.shape, (1,)), col_names=['input_size', 'output_size', 'num_params'])