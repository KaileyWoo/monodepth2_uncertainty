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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_path', type=str,
                        help='path to a model', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use', required=True)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--do_uncert", help="if set enables uncertainties", action="store_true")
    parser.add_argument("--do_monodepth2", help="if set enables do_monodepth2", action="store_true")

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print("begin\n")

    mono = True
    boot = False
    dropout = False
    nets = 8
    bootOnly = False
    dropoutOnly = False

    if mono:
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
        if args.do_monodepth2:
            depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        else:
            depth_decoder = networks.DepthUncertaintyDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4), uncert=True, dropout=False)
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)
        depth_decoder.to(device)
        depth_decoder.eval()

    ################################################################################################################################################
    if dropout:
        print("---------------------------------------Loading dropout models!")
        dropout_encoder_path = "./tmp/Baseline+AE_test/models/weights_19/encoder.pth"
        dropout_decoder_path = "./tmp/Baseline+AE_test/models/weights_19/depth.pth"
        dropout_encoder_dict = torch.load(dropout_encoder_path)
        if dropoutOnly:
            feed_height =dropout_encoder_dict['height']
            feed_width = dropout_encoder_dict['width']
        dropout_encoder = networks.ResnetEncoder(18, False)
        dropout_depth_decoder = networks.DepthUncertaintyDecoder(dropout_encoder.num_ch_enc, num_output_channels=1, uncert=True, dropout=True)
        model_dict = dropout_encoder.state_dict()
        dropout_encoder.load_state_dict({k: v for k, v in dropout_encoder_dict.items() if k in model_dict})
        dropout_depth_decoder.load_state_dict(torch.load(dropout_decoder_path))
        dropout_encoder.cuda()
        dropout_encoder.eval()
        dropout_depth_decoder.cuda()
        dropout_depth_decoder.eval()
    ################################################################################################################################################
    if boot:
        print("---------------------------------------Loading boot mono models!")
        boot_encoder_path = [os.path.join("./tmp/Bootstrap", "mono_model_boot%d"%i, "models/weights_19/encoder.pth") for i in range(nets)]
        boot_decoder_path = [os.path.join("./tmp/Bootstrap", "mono_model_boot%d"%i, "models/weights_19/depth.pth") for i in range(nets)]
        boot_encoder_dict = [torch.load(boot_encoder_path[i]) for i in range(nets)]
        if bootOnly:
            feed_height = boot_encoder_dict[0]['height']
            feed_width = boot_encoder_dict[0]['width']
        boot_encoder = [networks.ResnetEncoder(18, False) for i in range(nets)]
        boot_depth_decoder = []
        for i in range(nets):
            if i==1 or i==4 or i==6 or i==8 or i==9:
                boot_depth_decoder.append(networks.DepthUncertaintyDecoder(boot_encoder[i].num_ch_enc, num_output_channels=1, uncert=True, dropout=False))
            else:
                boot_depth_decoder.append(networks.DepthDecoder(boot_encoder[i].num_ch_enc, scales=range(4)))
        
        model_dict = [boot_encoder[i].state_dict() for i in range(nets)]
        for i in range(nets):
            boot_encoder[i].load_state_dict({k: v for k, v in boot_encoder_dict[i].items() if k in model_dict[i]})
            boot_depth_decoder[i].load_state_dict(torch.load(boot_decoder_path[i]))
            boot_encoder[i].cuda()
            boot_encoder[i].eval()
            boot_depth_decoder[i].cuda()
            boot_depth_decoder[i].eval()
    #################################################################################################################################################

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

    if not os.path.exists(os.path.join(output_directory, args.model_name)):
        os.makedirs(os.path.join(output_directory, args.model_name)) 

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
            print(original_width)
            print(original_height)
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            if args.do_monodepth2:
                features = encoder(input_image)
                outputs = depth_decoder(features)
                output = outputs[("disp", 0)]
            else:
                if mono:
                    outputs = depth_decoder(encoder(input_image))
                    output = outputs[("disp", 0)]
                    pred_uncert_A = torch.exp(outputs[("uncert", 0)]*0.0001)

                if boot:
                    disps_distribution = []
                    for i in range(nets):
                        outputs_boost =  boot_depth_decoder[i](boot_encoder[i](input_image))
                        disps_distribution.append( torch.unsqueeze(outputs_boost[("disp", 0)],0) )
                    disps_distribution =  torch.cat(disps_distribution, 0)   
                    pred_uncert_E = torch.var(disps_distribution, dim=0, keepdim=False)
                    if bootOnly:
                        output = torch.mean(disps_distribution, dim=0, keepdim=False)

                if dropout:
                    disps_distribution = []
                    uncert_distribution=[]
                    for i in range(nets):
                        outputs_dropout =  dropout_depth_decoder(dropout_encoder(input_image))
                        disps_distribution.append( torch.unsqueeze(outputs_dropout[("disp", 0)],0) )
                        uncert_distribution.append( torch.unsqueeze(outputs_dropout[("uncert", 0)],0) )
                    disps_distribution = torch.cat(disps_distribution, 0)
                    pred_uncert_E = torch.mean(torch.cat(uncert_distribution, 0), dim=0, keepdim=False)
                    #pred_uncert_E = torch.var(disps_distribution, dim=0, keepdim=False) + torch.mean(torch.cat(uncert_distribution, 0), dim=0, keepdim=False)
                    pred_uncert_E = torch.exp(pred_uncert_E)
                    if dropoutOnly:
                        output = torch.mean(disps_distribution, dim=0, keepdim=False)


            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            #name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            #scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            #np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized = torch.nn.functional.interpolate(output, (original_height, original_width), mode="bilinear", align_corners=False)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            name_dest_im = os.path.join(output_directory, args.model_name, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            # uncerts
            if args.do_uncert:
                if dropoutOnly or bootOnly:
                    uncert_resized =  torch.nn.functional.interpolate(pred_uncert_E, (original_height, original_width), mode="bilinear", align_corners=False)
                    uncert_resized_np = uncert_resized.squeeze().cpu().numpy()
                    uncert_resized_np = (uncert_resized_np - np.min(uncert_resized_np)) / (np.max(uncert_resized_np) - np.min(uncert_resized_np))
                    uncert_vmax = np.percentile(uncert_resized_np, 95)
                    uncert_normalizer = mpl.colors.Normalize(vmin=uncert_resized_np.min(), vmax=uncert_vmax)
                    uncert_mapper = cm.ScalarMappable(norm=uncert_normalizer, cmap='hot_r')   # hot_r
                    uncert_colormapped_im = (uncert_mapper.to_rgba(uncert_resized_np)[:, :, :3] *255).astype(np.uint8)
                    uncert_im = pil.fromarray(uncert_colormapped_im)
                    uncert_name_dest_im = os.path.join(output_directory, args.model_name, "{}_uncert_E.jpeg".format(output_name))
                    uncert_im.save(uncert_name_dest_im)
                else:
                    uncert_resized =  torch.nn.functional.interpolate(pred_uncert_A, (original_height, original_width), mode="bilinear", align_corners=False)
                    uncert_resized_np = uncert_resized.squeeze().cpu().numpy()
                    uncert_resized_np = (uncert_resized_np - np.min(uncert_resized_np)) / (np.max(uncert_resized_np) - np.min(uncert_resized_np))
                    uncert_vmax = np.percentile(uncert_resized_np, 95)
                    uncert_normalizer = mpl.colors.Normalize(vmin=uncert_resized_np.min(), vmax=uncert_vmax)
                    uncert_mapper = cm.ScalarMappable(norm=uncert_normalizer, cmap='hot')   # hot_r
                    uncert_colormapped_im = (uncert_mapper.to_rgba(uncert_resized_np)[:, :, :3] *255).astype(np.uint8)
                    uncert_im = pil.fromarray(uncert_colormapped_im)
                    uncert_name_dest_im = os.path.join(output_directory, args.model_name, "{}_uncert_A.jpeg".format(output_name))
                    uncert_im.save(uncert_name_dest_im)

                    if boot or dropout:
                        uncert_resized =  torch.nn.functional.interpolate(pred_uncert_E, (original_height, original_width), mode="bilinear", align_corners=False)
                        uncert_resized_np = uncert_resized.squeeze().cpu().numpy()
                        uncert_resized_np = (uncert_resized_np - np.min(uncert_resized_np)) / (np.max(uncert_resized_np) - np.min(uncert_resized_np))
                        uncert_vmax = np.percentile(uncert_resized_np, 95)
                        uncert_normalizer = mpl.colors.Normalize(vmin=uncert_resized_np.min(), vmax=uncert_vmax)
                        uncert_mapper = cm.ScalarMappable(norm=uncert_normalizer, cmap='hot')   # hot_r
                        uncert_colormapped_im = (uncert_mapper.to_rgba(uncert_resized_np)[:, :, :3] *255).astype(np.uint8)
                        uncert_im = pil.fromarray(uncert_colormapped_im)
                        uncert_name_dest_im = os.path.join(output_directory, args.model_name, "{}_uncert_E.jpeg".format(output_name))
                        uncert_im.save(uncert_name_dest_im)
                
            
            print("   Processed {:d} of {:d} images - saved prediction to {}".format(idx + 1, len(paths), name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
