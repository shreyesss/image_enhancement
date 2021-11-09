import torch
import torch.nn as nn
import torch.nn.functional as F


class MonodepthLoss(nn.modules.Module):
    def __init__(self, SSIM_w=0.1, L1_weight = 0.9,do_stereo=True):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.L1_weight = L1_weight
        self.L1_loss = torch.nn.SmoothL1Loss()
        self.do_stereo = do_stereo


    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.mean(torch.clamp((1 - SSIM) / 2, 0, 1))


    def forward(self, inputs,outputs):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
          
        Return:
            (float): The loss
        """
        if self.do_stereo == True :
           
            img_left = inputs[:,:3,:,:]
            img_right = inputs[:,3:,:,:]
            gt_left = outputs[:,:3,:,:]
            gt_right = outputs[:,3:,:,:]
            ssim_loss = 0.5 * (self.SSIM(img_left,gt_left) + self.SSIM(img_right,gt_right))
            L1_loss = 0.5 * (self.L1_loss(img_left,gt_left) + self.L1_loss(img_right,gt_right))
            loss = 0.9 * L1_loss + 0.1 * ssim_loss
        else :
            ssim_loss = self.SSIM(inputs,outputs)
            L1_loss = self.L1_loss(inputs,outputs)
            loss = 0.9 * L1_loss + 0.1 * ssim_loss

        return loss
