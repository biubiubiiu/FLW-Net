import argparse
import os
import os.path as osp
import random
from enum import Enum

import numpy as np
import torch
import torchvision.transforms.functional as T
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_mean', type=float, default=0.8, help='expected mean value.')
    parser.add_argument('--use_gt_mean', action='store_true', help='Use ground-truth mean values')
    parser.add_argument('--data_path', type=str, default='./LOL/', help='path to dataset')
    parser.add_argument('--save_path', type=str, default='./result', help='path to save checkpoints and logs')
    parser.add_argument('--num_epoch', type=int, default=30000, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size')
    parser.add_argument('--patch_size', type=int, default=100, help='size of image patches for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--pad_multiple_to', type=int, default=32)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--eval_step', type=int, default=30)
    parser.add_argument('--save_step', type=int, default=1000)
    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    parser.add_argument('--ckpt', type=str, default=None, help='path to checkpoint')
    return init_args(parser.parse_args())


def init_args(args):
    if args.phase == 'test' and args.ckpt is None:
        assert 'checkpoint should be specified in the test phase'

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(osp.join(args.save_path, 'checkpoints'), exist_ok=True)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return args


class LOLDataset(Dataset):
    def __init__(self, data_path, input_dir='low', gt_dir='high', training=True, patch_size=128):
        super(LOLDataset, self).__init__()

        self.dataroot = data_path
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.fns = os.listdir(osp.join(data_path, gt_dir))
        self.training = training
        self.patch_size = patch_size

    def __getitem__(self, i):
        fn = self.fns[i]
        input_path = osp.join(self.dataroot, self.input_dir, fn)
        target_path = osp.join(self.dataroot, self.gt_dir, fn)

        input = T.to_tensor(Image.open(input_path))
        target = T.to_tensor(Image.open(target_path))

        if self.training:
            i, j, th, tw = RandomCrop.get_params(input, (self.patch_size, self.patch_size))
            input = T.crop(input, i, j, th, tw)
            target = T.crop(target, i, j, th, tw)

        return {'input': input, 'target': target, 'fn': fn}

    def __len__(self):
        return len(self.fns)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    class Summary(Enum):
        NONE = 0
        AVERAGE = 1
        SUM = 2
        COUNT = 3

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is self.Summary.NONE:
            fmtstr = ''
        elif self.summary_type is self.Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is self.Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is self.Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError(f'invalid summary type {self.summary_type}')

        return fmtstr.format(**self.__dict__)
