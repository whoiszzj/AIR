import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(self.channel)

    def _gaussian(self):
        gauss = torch.Tensor([exp(-(x - self.window_size//2)**2/float(2*1.5**2)) for x in range(self.window_size)])
        return gauss/gauss.sum()

    def _create_window(self, channel):
        _1D_window = self._gaussian().unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous())
        return window

    def _ssim(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding = self.window_size//2, groups = self.channel)
        mu2 = F.conv2d(img2, self.window, padding = self.window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, self.window, padding = self.window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding = self.window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding = self.window_size//2, groups = self.channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return self._ssim(img1, img2)

class RGBLoss(nn.Module):
    def __init__(self, lambda_val=0.7):
        super(RGBLoss, self).__init__()
        self.lambda_val = lambda_val
        self.ssim = SSIM(size_average=False)

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction='none')
        mse_loss = mse_loss.mean(dim=tuple(range(1, pred.dim())))

        ssim_loss = 1 - self.ssim(pred, target)
        ssim_loss = ssim_loss.mean(dim=tuple(range(1, pred.dim())))

        loss = self.lambda_val * mse_loss + (1 - self.lambda_val) * ssim_loss
        return loss # [B]
    