from __future__ import absolute_import, division, print_function
import warnings

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

import kitti_utils as kitti_utils
from layers import *
from utils import *
from options import MonodepthOptions
import datasets as datasets
import networks as legacy
import networks
import progressbar
import matplotlib.pyplot as plt

import networks

import sys

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Real-world scale factor (see Monodepth2)
STEREO_SCALE_FACTOR = 5.4
uncertainty_metrics = ["abs_rel", "rmse", "a1"]

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def compute_eigen_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_eigen_errors_v2(gt, pred, metrics=uncertainty_metrics, mask=None, reduce_mean=False):
    """Revised compute_eigen_errors function used for uncertainty metrics, with optional reduce_mean argument and (1-a1) computation
    """
    results = []
    
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    
    if "abs_rel" in metrics:
        abs_rel = (np.abs(gt - pred) / gt)
        if reduce_mean:
            abs_rel = abs_rel.mean()
        results.append(abs_rel)

    if "rmse" in metrics:
        rmse = (gt - pred) ** 2
        if reduce_mean:
            rmse = np.sqrt(rmse.mean())
        results.append(rmse)

    if "a1" in metrics:
        a1 = np.maximum((gt / pred), (pred / gt))
        if reduce_mean:
        
            # invert to get outliers
            a1 = (a1 >= 1.25).mean()
        results.append(a1)

    return results

def compute_aucs(gt, pred, uncert, intervals=50):
    """Computation of auc metrics
    """
    
    # results dictionaries
    AUSE = {"abs_rel":0, "rmse":0, "a1":0}
    AURG = {"abs_rel":0, "rmse":0, "a1":0}

    # revert order (high uncertainty first)
    uncert = -uncert
    true_uncert = compute_eigen_errors_v2(gt,pred)
    true_uncert = {"abs_rel":-true_uncert[0],"rmse":-true_uncert[1],"a1":-true_uncert[2]}

    # prepare subsets for sampling and for area computation
    quants = [100./intervals*t for t in range(0,intervals)]
    plotx = [1./intervals*t for t in range(0,intervals+1)]

    # get percentiles for sampling and corresponding subsets
    thresholds = [np.percentile(uncert, q) for q in quants]
    subs = [(uncert >= t) for t in thresholds]

    # compute sparsification curves for each metric (add 0 for final sampling)
    sparse_curve = {m:[compute_eigen_errors_v2(gt,pred,metrics=[m],mask=sub,reduce_mean=True)[0] for sub in subs]+[0] for m in uncertainty_metrics }

    # human-readable call
    '''
    sparse_curve =  {"rmse":[compute_eigen_errors_v2(gt,pred,metrics=["rmse"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0], 
                     "a1":[compute_eigen_errors_v2(gt,pred,metrics=["a1"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0],
                     "abs_rel":[compute_eigen_errors_v2(gt,pred,metrics=["abs_rel"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0]}
    '''
    
    # get percentiles for optimal sampling and corresponding subsets
    opt_thresholds = {m:[np.percentile(true_uncert[m], q) for q in quants] for m in uncertainty_metrics}
    opt_subs = {m:[(true_uncert[m] >= o) for o in opt_thresholds[m]] for m in uncertainty_metrics}

    # compute sparsification curves for optimal sampling (add 0 for final sampling)
    opt_curve = {m:[compute_eigen_errors_v2(gt,pred,metrics=[m],mask=opt_sub,reduce_mean=True)[0] for opt_sub in opt_subs[m]]+[0] for m in uncertainty_metrics}

    # compute metrics for random sampling (equal for each sampling)
    rnd_curve = {m:[compute_eigen_errors_v2(gt,pred,metrics=[m],mask=None,reduce_mean=True)[0] for t in range(intervals+1)] for m in uncertainty_metrics}    

    # compute error and gain metrics
    for m in uncertainty_metrics:

        # error: subtract from method sparsification (first term) the oracle sparsification (second term)
        AUSE[m] = np.trapz(sparse_curve[m], x=plotx) - np.trapz(opt_curve[m], x=plotx)
        
        # gain: subtract from random sparsification (first term) the method sparsification (second term)
        AURG[m] = rnd_curve[m][0] - np.trapz(sparse_curve[m], x=plotx)

    # returns a dictionary with AUSE and AURG for each metric
    return {m:[AUSE[m], AURG[m]] for m in uncertainty_metrics}

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    
    print("-> Beginning inference...")
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"), False)
    ################################################################################################################################################
    mono = True
    boot = False
    dropout = False
    nets = 8
    bootOnly = False
    dropoutOnly = False

    if mono:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)
        print("-> Loading weights from {}".format(opt.load_weights_folder))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path)
        height = encoder_dict['height']
        width = encoder_dict['width']
        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthUncertaintyDecoder(encoder.num_ch_enc, num_output_channels=1, uncert=True, dropout=False)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
    #################################################################################################################################################
    if dropout:
        print("---------------------------------------Loading dropout models!")
        dropout_encoder_path = "./tmp/Dropout/models/weights_19/encoder.pth"
        dropout_decoder_path = "./tmp/Dropout/models/weights_19/depth.pth"
        dropout_encoder_dict = torch.load(dropout_encoder_path)
        if dropoutOnly:
            height =dropout_encoder_dict['height']
            width = dropout_encoder_dict['width']
        dropout_encoder = networks.ResnetEncoder(opt.num_layers, False)
        dropout_depth_decoder = networks.DepthUncertaintyDecoder(dropout_encoder.num_ch_enc, num_output_channels=1, uncert=True, dropout=True)
        model_dict = dropout_encoder.state_dict()
        dropout_encoder.load_state_dict({k: v for k, v in dropout_encoder_dict.items() if k in model_dict})
        dropout_depth_decoder.load_state_dict(torch.load(dropout_decoder_path))
        dropout_encoder.cuda()
        dropout_encoder.eval()
        dropout_depth_decoder.cuda()
        dropout_depth_decoder.eval()
    #################################################################################################################################################
    if boot:
        print("---------------------------------------Loading separate boot mono models!")
        boot_encoder_path = [os.path.join("./tmp/Bootstrap", "mono_model_boot%d"%i, "models/weights_19/encoder.pth") for i in range(nets)]
        boot_decoder_path = [os.path.join("./tmp/Bootstrap", "mono_model_boot%d"%i, "models/weights_19/depth.pth") for i in range(nets)]
        boot_encoder_dict = [torch.load(boot_encoder_path[i]) for i in range(nets)]
        if bootOnly:
            height = boot_encoder_dict[0]['height']
            width = boot_encoder_dict[0]['width']
        boot_encoder = [networks.ResnetEncoder(18, False) for i in range(nets)]
        boot_depth_decoder = []
        for i in range(nets):
            if i==1 or i==4 or i==6 or i==8 or i==9:
                boot_depth_decoder.append(networks.DepthUncertaintyDecoder(boot_encoder[i].num_ch_enc, num_output_channels=1, uncert=True, dropout=False))
            else:
                boot_depth_decoder.append(networks.DepthDecoder(boot_encoder[i].num_ch_enc, opt.scales))
        model_dict = [boot_encoder[i].state_dict() for i in range(nets)]
        for i in range(nets):
            boot_encoder[i].load_state_dict({k: v for k, v in boot_encoder_dict[i].items() if k in model_dict[i]})
            boot_depth_decoder[i].load_state_dict(torch.load(boot_decoder_path[i]))
            boot_encoder[i].cuda()
            boot_encoder[i].eval()
            boot_depth_decoder[i].cuda()
            boot_depth_decoder[i].eval()
    #################################################################################################################################################

    img_ext = '.png' if opt.png else '.jpg'
    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,height, width, [0], 4, is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,pin_memory=True, drop_last=False)

    # accumulators for depth and uncertainties
    pred_disps = []
    pred_uncerts = []

    print("-> Computing predictions with size {}x{}".format(width, height))
    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()
            if opt.post_process:
                # post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            if mono:
                output = depth_decoder(encoder(input_color))
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_uncert_A = torch.exp(output[("uncert", 0)]).cpu()[0].numpy()

            if dropout:
                disps_distribution = []
                for i in range(nets):
                    outputs_dropout =  dropout_depth_decoder(dropout_encoder(input_color))
                    disps_distribution.append( torch.unsqueeze(outputs_dropout[("disp", 0)],0) )
                disps_distribution = torch.cat(disps_distribution, 0)
                pred_uncert_E = torch.var(disps_distribution, dim=0, keepdim=False).cpu()[0].numpy()
                if dropoutOnly:
                    output = torch.mean(disps_distribution, dim=0, keepdim=False)
                    pred_disp, _ = disp_to_depth(output, opt.min_depth, opt.max_depth)       

            if boot:
                disps_distribution = []
                for i in range(nets):
                    outputs_boost = boot_depth_decoder[i](boot_encoder[i](input_color))
                    disps_distribution.append( torch.unsqueeze(outputs_boost[("disp", 0)],0) )
                disps_distribution = torch.cat(disps_distribution, 0)
                pred_uncert_E = torch.var(disps_distribution, dim=0, keepdim=False).cpu()[0].numpy()
                if bootOnly:
                    output = torch.mean(disps_distribution, dim=0, keepdim=False)
                    pred_disp, _ = disp_to_depth(output, opt.min_depth, opt.max_depth) 
                
            if dropoutOnly or bootOnly:
                pred_uncert = pred_uncert_E
            else:
                pred_uncert = pred_uncert_A
                          
                
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process: 
                N = pred_disp.shape[0] // 2
                pred_uncert = np.abs(pred_disp[:N] - pred_disp[N:, :, ::-1])		
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                pred_uncerts.append(pred_uncert)

            pred_disps.append(pred_disp)

            pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
            pred_uncerts.append(pred_uncert)

    pred_disps = np.concatenate(pred_disps)
    pred_uncerts = np.concatenate(pred_uncerts)

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

#########################################################################################################################################################################
    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")
    
    errors = []
    ratios = []

    aucs = {"abs_rel":[], "rmse":[], "a1":[]}
    
    for i in range(len(gt_depths)):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        pred_uncert = pred_uncerts[i]
        pred_uncert = cv2.resize(pred_uncert, (gt_width, gt_height))

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = (gt_depth > 0)

        # apply masks
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        pred_uncert = pred_uncert[mask]

        # apply scale factor and depth cap
        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        # get Eigen's metrics
        errors.append(compute_eigen_errors(gt_depth, pred_depth))
        # get uncertainty metrics (AUSE and AURG)
        scores = compute_aucs(gt_depth, pred_depth, pred_uncert)
        # append AUSE and AURG to accumulators
        [aucs[m].append(scores[m]) for m in uncertainty_metrics ]
    
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

     # compute mean depth metrics and print
    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    # compute mean uncertainty metrics and print
    for m in uncertainty_metrics:
        aucs[m] = np.array(aucs[m]).mean(0)

    print("\n  " + ("{:>8} | " * 6).format("abs_rel", "", "rmse", "", "a1", ""))
    print("  " + ("{:>8} | " * 6).format("AUSE", "AURG", "AUSE", "AURG", "AUSE", "AURG"))
    print(("&{:8.3f}  " * 6).format(*list(aucs["abs_rel"])+list(aucs["rmse"])+list(aucs["a1"])) + "\\\\")

    # see you next time! 
    print("\n-> Done!")

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    options = MonodepthOptions()
    evaluate(options.parse())
