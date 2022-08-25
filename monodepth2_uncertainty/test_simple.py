# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

import matplotlib.pyplot as plt

import cv2


def get_mono_ratio(disp, gt):
    """Returns the median scaling factor
    """
    mask = gt>0
    return np.median(gt[mask]) / np.median(cv2.resize(1/disp, (gt.shape[1], gt.shape[0]))[mask])

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_path', type=str,
                        help='path to a model', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    #assert args.model_name is not None, \
     #   "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #download_model_if_doesnt_exist(args.model_name)
    #model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", args.model_path)
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    depth_decoder_path = os.path.join(args.model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    #depth_decoder = networks.DepthUncertaintyDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4), uncert=True, dropout=True)

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            #disp = outputs[("disp", 0)]
            #disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

            '''disps_distribution = []
            for i in range(8):
                output = depth_decoder(encoder(input_image))
                disps_distribution.append( torch.unsqueeze(output[("disp", 0)],0) )
            disps_distribution = torch.cat(disps_distribution, 0)
            pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False).cpu()[0].numpy()
            output = torch.mean(disps_distribution, dim=0, keepdim=False)
            pred_disp, _ = disp_to_depth(output, 0.1, 100)
            pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
            pred_uncert = np.concatenate(pred_uncert)
            disp_resized = torch.nn.functional.interpolate(pred_disp, (original_height, original_width), mode="bilinear", align_corners=False)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disp = np.concatenate(pred_disp)'''

            # save uncertainties
            #cv2.imwrite(os.path.join(output_directory, 'raw_uncert_10.png'), (pred_uncert*(256*256-1)).astype(np.uint16))

            # save colored depth maps
            #plt.imsave(os.path.join(output_directory, 'qual_disp_10.png'), pred_disp, cmap='magma')

            # save colored uncertainty maps
            #plt.imsave(os.path.join(output_directory, 'qual_uncert_10.png'), pred_uncert, cmap='hot')

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            #name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            #scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            #np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            pred_disp, _ = disp_to_depth(outputs[("disp", 0)], 0.1, 100)
            disp_resized = torch.nn.functional.interpolate(pred_disp, (original_height, original_width), mode="bilinear", align_corners=False)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            # uncerts
            ##uncert = uncert.squeeze().cpu().numpy()
            ##uncert = (uncert - np.min(uncert)) / (np.max(uncert) - np.min(uncert))
            ##plt.imsave(os.path.join(output_directory,  "{}_uncert.jpeg".format(output_name)), uncert, cmap='hot')
            '''
            #uncert = outputs[("uncert", 0)]
            #uncert = torch.exp(outputs[("uncert", 0)])  #.cpu()[:, 0].numpy()
            uncert_resized =  torch.nn.functional.interpolate(uncert, (original_height, original_width), mode="bilinear", align_corners=False)

            uncert_resized_np = uncert_resized.squeeze().cpu().numpy()
            uncert_resized_np = (uncert_resized_np - np.min(uncert_resized_np)) / (np.max(uncert_resized_np) - np.min(uncert_resized_np))
            uncert_vmax = np.percentile(uncert_resized_np, 95)
            uncert_normalizer = mpl.colors.Normalize(vmin=uncert_resized_np.min(), vmax=uncert_vmax)
            uncert_mapper = cm.ScalarMappable(norm=uncert_normalizer, cmap='hot_r')   # hot_r
            uncert_colormapped_im = (uncert_mapper.to_rgba(uncert_resized_np)[:, :, :3] *255).astype(np.uint8)
            uncert_im = pil.fromarray(uncert_colormapped_im)

            uncert_name_dest_im = os.path.join(output_directory, "{}_uncert.jpeg".format(output_name))
            uncert_im.save(uncert_name_dest_im)
            '''
            #print("   Processed {:d} of {:d} images - saved prediction to {}".format(
           #    idx + 1, len(paths), name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
