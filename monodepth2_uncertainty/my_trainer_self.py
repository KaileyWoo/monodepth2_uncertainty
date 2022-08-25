from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.models = {}
        self.parameters_to_train = []
        self.boot_encoder = []
        self.boot_depth_decoder = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)     #default=[0, 1, 2, 3]
        self.num_input_frames = len(self.opt.frame_ids)     #default=[0, -1, 1]
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames      #default="pairs"

        self. load_depth_pose_models()

        #self.nets = 8
        #self.load_boot_models()   # load boots net
        self.load_teacher_models()

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.load_data()

        if self.opt.load_weights_folder is not None:
            self.load_model()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        #self.load_other_layers()    # 加载训练SSIM loss 层、深度图转点云层（return cam_points）、3D点云转相机坐标（return pix_coords）
        
        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            self.model_optimizer.zero_grad() #清空梯度
            #accumulation_steps = 3
            outputs, losses = self.process_batch(inputs)
            losses["loss"].backward()    #反向传播，计算当前梯度
      
            #使用梯度累加
            #if((batch_idx+1)%accumulation_steps == 0):
            self.model_optimizer.step()    #一次性更新权重
            #self.model_optimizer.zero_grad() #清空梯度
        
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                #self.compute_depth_losses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        self.T_outputs = self.T_decoder(self.T_encoder(inputs["color_aug", 0, 0]))
        outputs = self.models["depth"](self.models["encoder"](inputs["color_aug", 0, 0]))
        
        ''' load dropout teacher'''
        '''disps_distribution_0 = []
        disps_distribution_1 = []
        disps_distribution_2 = []
        disps_distribution_3 = []
        for i in range(8):
            dropout_outputs = self.T_decoder(self.T_encoder(inputs["color_aug", 0, 0]))
            disps_distribution_0.append( torch.unsqueeze(dropout_outputs[("disp", 0)],0) )
            disps_distribution_1.append( torch.unsqueeze(dropout_outputs[("disp", 1)],0) )
            disps_distribution_2.append( torch.unsqueeze(dropout_outputs[("disp", 2)],0) )
            disps_distribution_3.append( torch.unsqueeze(dropout_outputs[("disp", 3)],0) )

        outputs[("dropout_disp", 0)] = torch.mean(torch.cat(disps_distribution_0, 0), dim=0, keepdim=False)
        outputs[("dropout_disp", 1)] = torch.mean(torch.cat(disps_distribution_1, 0), dim=0, keepdim=False)
        outputs[("dropout_disp", 2)] = torch.mean(torch.cat(disps_distribution_2, 0), dim=0, keepdim=False)
        outputs[("dropout_disp", 3)] = torch.mean(torch.cat(disps_distribution_3, 0), dim=0, keepdim=False)

        outputs[("dropout_uncert", 0)] = torch.var(torch.cat(disps_distribution_0, 0), dim=0, keepdim=False)
        outputs[("dropout_uncert", 1)] = torch.var(torch.cat(disps_distribution_1, 0), dim=0, keepdim=False)
        outputs[("dropout_uncert", 2)] = torch.var(torch.cat(disps_distribution_2, 0), dim=0, keepdim=False)
        outputs[("dropout_uncert", 3)] = torch.var(torch.cat(disps_distribution_3, 0), dim=0, keepdim=False)

        del disps_distribution_0
        del disps_distribution_1
        del disps_distribution_2
        del disps_distribution_3'''

        
        '''load boot teacher'''
        '''disps_distribution_0 = []
        disps_distribution_1 = []
        disps_distribution_2 = []
        disps_distribution_3 = []
        for i in range(self.nets):
            outputs_boost =  self.boot_depth_decoder[i](self.boot_encoder[i](inputs["color_aug", 0, 0]))
            disps_distribution_0.append( torch.unsqueeze(outputs_boost[("disp", 0)],0) )
            disps_distribution_1.append( torch.unsqueeze(outputs_boost[("disp", 1)],0) )
            disps_distribution_2.append( torch.unsqueeze(outputs_boost[("disp", 2)],0) )
            disps_distribution_3.append( torch.unsqueeze(outputs_boost[("disp", 3)],0) )

        #outputs[("boot_disp", 0)] = torch.mean(torch.cat(disps_distribution_0, 0), dim=0, keepdim=False)
        #outputs[("boot_disp", 1)] = torch.mean(torch.cat(disps_distribution_1, 0), dim=0, keepdim=False)
        #outputs[("boot_disp", 2)] = torch.mean(torch.cat(disps_distribution_2, 0), dim=0, keepdim=False)
        #outputs[("boot_disp", 3)] = torch.mean(torch.cat(disps_distribution_3, 0), dim=0, keepdim=False)

        outputs[("boot_uncert", 0)] = torch.var(torch.cat(disps_distribution_0, 0), dim=0, keepdim=False)
        outputs[("boot_uncert", 1)] = torch.var(torch.cat(disps_distribution_1, 0), dim=0, keepdim=False)
        outputs[("boot_uncert", 2)] = torch.var(torch.cat(disps_distribution_2, 0), dim=0, keepdim=False)
        outputs[("boot_uncert", 3)] = torch.var(torch.cat(disps_distribution_3, 0), dim=0, keepdim=False)

        del disps_distribution_0
        del disps_distribution_1
        del disps_distribution_2
        del disps_distribution_3'''
    
        #outputs.update(self.predict_poses(inputs))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # In this setting, we compute the pose to each source frame via a separate forward pass through the pose network.
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs
            del outputs
            del losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        # outputs["disp"]直接输出的就是视差图，并且仍然多尺度[0,1,2,3]分布。
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            #source_scale = 0

            # 将disp值映射到[0.01,10]，并求倒数就能得到深度值
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            # 将深度值存放到outputs["depth"...]中
            outputs[("depth", 0, scale)] = depth

            '''for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]     # 通过pose net得到的输出，用于将3纬点云投影成二维图像

                # 将深度图投影成3维点云
                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                # 将3维点云投影成二维图像
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)  
                # 将二维图像赋值给outputs[("sample"..)]
                outputs[("sample", frame_id, scale)] = pix_coords

                # outputs上某点(x,y)的三个通道像素值来自于inputs上的(x',y')
                # 而x'和y'则由outputs(x,y)的最低维[0]和[1]
                outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)],outputs[("sample", frame_id, scale)],padding_mode="border")

                outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]'''

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)  # abs 绝对值
        l1_loss = abs_diff.mean(1, True)
        #abs_diff1  = abs_diff*abs_diff
        #l2_loss = abs_diff1.mean(1, True)

        '''if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss'''

        #return reprojection_loss
        return l1_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        # 按尺度来计算loss
        #for scale in self.opt.scales:
        ########################################################################3
        scale = 0
        loss = 0
        L1_losses = []
        #source_scale = 0
        
        color = inputs[("color", 0, scale)]
        #target = inputs[("color", 0, scale)]

        uncert_A_flag = True
        uncert_E_flag = False 
        
        T_disp = self.T_outputs[("disp", scale)]
        #T_disp = outputs[("boot_disp", scale)]
        #T_disp = outputs[("dropout_disp", 0)]
        disp = outputs[("disp", scale)]
        uncert_A = torch.exp(outputs[("uncert", scale)])

        '''  这里2选一'''
        if uncert_E_flag:
            uncert_E =  torch.exp(outputs[("boot_uncert", scale)])
            uncert_E =  torch.exp(outputs[("dropout_uncert", scale)])

        L1_loss = self.compute_reprojection_loss(disp, T_disp)
        L1_losses.append(L1_loss)
        if uncert_A_flag:
            uncert_loss1 = L1_loss/(uncert_A  + 1e-10) + uncert_A 
            L1_losses.append(uncert_loss1)
        if uncert_E_flag:
            uncert_loss2 = L1_loss/(uncert_E + 1e-10) + uncert_E
            L1_losses.append(uncert_loss2)
        
        L1_losses = torch.cat(L1_losses, 1)  #按维数1拼接（横着拼）
        loss += L1_losses.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)

        loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
    
        total_loss += loss
    
        losses["loss/{}".format(scale)] = loss
        ##################################################################################################
        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                writer.add_image("color_{}_{}/{}".format(0, s, j),inputs[("color", 0, s)][j].data, self.step)
                writer.add_image("disp_{}/{}".format(s, j),normalize_image(outputs[("disp", s)][j]), self.step)
                #writer.add_image("dropout_uncert_{}/{}".format(s, j),normalize_image(torch.exp(outputs[("dropout_uncert", s)][j])), self.step) 
                writer.add_image("uncert_{}/{}".format(s, j),normalize_image(torch.exp(outputs[("uncert", s)][j])), self.step) 
                #writer.add_image("ALL_uncert_{}/{}".format(s, j),normalize_image( torch.exp(outputs[("dropout_uncert", s)] + outputs[("uncert", s)][j])), self.step) 
                #writer.add_image("dropout_uncert_{}/{}".format(s, j),normalize_image(torch.exp(outputs[("dropout_uncert", s)][j])), self.step)         
                #writer.add_image("automask_{}/{}".format(s, j),outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
               

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def load_teacher_models(self):
        baseline = False
        uncertainty = True
        dropout = False

        if baseline:
            encoder_path = "./tmp//models/weights_19/encoder.pth"
            decoder_path = "./tmp/model_fullData_1026/models/weights_19/depth.pth"

        if dropout:
            encoder_path = "./tmp/droupout_T_model/models/weights_19/encoder.pth"
            decoder_path = "./tmp/droupout_T_model/models/weights_19/depth.pth"

        if uncertainty:
            print("load Baseline_A!\n")
            encoder_path = "./tmp/Baseline_A/models/weights_19/encoder.pth"
            decoder_path = "./tmp/Baseline_A/models/weights_19/depth.pth"

        encoder_dict = torch.load(encoder_path)
        print("-------------------------------------load teacher encoder and decoder! ")
       	self.T_encoder = networks.ResnetEncoder(18, False)
        if baseline:
            self.T_decoder = networks.DepthDecoder(self.T_encoder.num_ch_enc, self.opt.scales)
        else:
            self.T_decoder  = networks.DepthUncertaintyDecoder( self.T_encoder.num_ch_enc, num_output_channels=1, uncert=uncertainty, dropout=dropout)
        
        model_dict = self.T_encoder.state_dict() 
        self.T_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        self.T_decoder.load_state_dict(torch.load(decoder_path))
        self.T_encoder.cuda()
        self.T_encoder.eval()
        self.T_decoder.cuda()
        self.T_decoder.eval()

    def load_boot_models(self):
        print("---------------------------------------Loading  boot teacher mono models!")
        boot_encoder_path = [os.path.join("./tmp/boots", "mono_model_boot%d"%i, "models/weights_19/encoder.pth") for i in range(self.nets)]
        boot_decoder_path = [os.path.join("./tmp/boots", "mono_model_boot%d"%i, "models/weights_19/depth.pth") for i in range(self.nets)]
        boot_encoder_dict = [torch.load(boot_encoder_path[i]) for i in range(self.nets)]
    
        self.boot_encoder = [networks.ResnetEncoder(18, False) for i in range(self.nets)]
        for i in range(self.nets):
            if i==1 or i==4 or i==6 or i==8 or i==9:
                self.boot_depth_decoder.append(networks.DepthUncertaintyDecoder( self.boot_encoder[i].num_ch_enc, num_output_channels=1, uncert=True, dropout=False))
            else:
                self.boot_depth_decoder.append(networks.DepthDecoder( self.boot_encoder[i].num_ch_enc, self.opt.scales))
        model_dict = [ self.boot_encoder[i].state_dict() for i in range(self.nets)]
        for i in range(self.nets):
             self.boot_encoder[i].load_state_dict({k: v for k, v in boot_encoder_dict[i].items() if k in model_dict[i]})
             self.boot_depth_decoder[i].load_state_dict(torch.load(boot_decoder_path[i]))
             self.boot_encoder[i].cuda()
             self.boot_encoder[i].eval()
             self.boot_depth_decoder[i].cuda()
             self.boot_depth_decoder[i].eval()

    def load_depth_pose_models(self):
        print("-------------------------------------Load depth & pose model!")
        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        #self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"] = networks.DepthUncertaintyDecoder(self.models["encoder"].num_ch_enc, self.opt.scales, uncert = True, dropout = False)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        '''self.models["pose_encoder"] = networks.ResnetEncoder(self.opt.num_layers,self.opt.weights_init == "pretrained",num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1,num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())'''

    def load_data(self):
        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,"mydata": datasets.MYRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"), True)
        val_filenames = readlines(fpath.format("val"), True)
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(self.opt.data_path, train_filenames, self.opt.height, self.opt.width,self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader( train_dataset, self.opt.batch_size, True,num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(self.opt.data_path, val_filenames, self.opt.height, self.opt.width,self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, True,num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))

    def load_other_layers(self):
        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)






       





