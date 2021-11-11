import torch
import torch.nn as nn
import numpy as np
#import pandas as pd
import os
import  cv2
#from collections import Counter
#import pickle


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3



width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

# def load_gt_disp_kitti(path):
#     gt_disparities = []
#     for i in range(200):
#         disp = cv2.imread(path + "/training/disp_noc_0/" + str(i).zfill(6) + "_10.png", -1)
#         disp = disp.astype(np.float32) / 256
#         gt_disparities.append(disp)
#     return gt_disparities

def convert_disps_to_depths_kitti(gt_disparities, pred_disparities):
    # gt_depths = []
    # pred_depths = []
    # pred_disparities_resized = []
    
    #for i in range(len(gt_disparities)):
    gt_disp = gt_disparities
    height, width = gt_disp.shape

    pred_disp = pred_disparities
    pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

    pred_disparities_resized = pred_disp 

    mask = gt_disp > 0

    gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
    pred_depth = width_to_focal[width] * 0.54 / pred_disp
    #print(gt_depth.shape,np.max(gt_depth))
    #print(pred_depth.shape,np.max(pred_depth))


    return gt_depth, pred_depth, pred_disparities_resized


def get_PSNR(pred,out):
    mse = torch.mean((pred - out) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def get_SSIM(x,y):
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

        return torch.mean(torch.clamp(SSIM, 0, 1))




def compute_metrics(pred,out):

    N,C,H,W = pred.shape
    metrics = {}
    metrics["PSNR"] = 0
    metrics["SSIM"] = 0
    for  i in range(N):
        #print(gt.shape)
        # print(disp.shape)
        # print(gt[i,0,:].shape)
        pred_left = pred[i,:3,:,:]
        gt_left = out[i,:3,:,:]
        pred_right = pred[i,3:,:,:]
        gt_right = out[i,3:,:,:]

        PSNR = 0.5 * (get_PSNR(pred_left,gt_left)  + get_PSNR(pred_right,gt_right))
        SSIM = 0.5* (get_SSIM(pred_left,gt_left) + get_SSIM(pred_right,gt_right))

    
        metrics["PSNR"] += PSNR
        metrics["SSIM"] += SSIM
       

    metrics["N"]= N
    return metrics

    