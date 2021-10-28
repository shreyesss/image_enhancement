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



def compute_metrics(disp,gt):

    N,C,H,W = disp.shape
    metrics = {}
    metrics["d1_all"] = 0
    metrics["abs_rel"] = 0
    metrics["sq_rel"] = 0
    metrics["rms"] = 0 
    metrics["log_rms"] = 0
    metrics["a1"] = 0
    metrics["a2"] = 0
    metrics["a3"] = 0
    metrics["N"] = 0
    for  i in range(N):
        #print(gt.shape)
        # print(disp.shape)
        # print(gt[i,0,:].shape)
        pred_disp = disp[i,0,:,:]
        gt_disp = gt[i,0,:,:]

        gt_depth, pred_depth, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disp,pred_disp)

        mask = gt_disp > 0
        pred_disp = pred_disparities_resized

        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        metrics["d1_all"] = metrics["d1_all"] +  100.0 * bad_pixels.sum() / mask.sum()
        mets = compute_errors(gt_depth[mask], pred_depth[mask])
        metrics["abs_rel"] = metrics["abs_rel"] + mets[0]
        metrics["sq_rel"]  = metrics["sq_rel"]  + mets[1]
        metrics["rms"] = metrics["rms"] + mets[2]
        metrics["log_rms"] = metrics["log_rms"] + mets[3]
        metrics["a1"] = metrics["a1"] + mets[4]
        metrics["a2"] = metrics["a2"] + mets[5]
        metrics["a3"] = metrics["a3"] + mets[6]

    metrics["N"]= N
    return metrics

    