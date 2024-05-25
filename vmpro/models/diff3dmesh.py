from os import path as osp
import numpy as np
np.set_printoptions(suppress=True)
from functools import partial
import time
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict
from termcolor import colored
import logging

from vmpro.core.config import cfg
from vmpro.models import simple3dpose
from vmpro.models.simple3dpose import norm_heatmap
from vmpro.utils.funcs_utils import load_checkpoint
from vmpro.utils.diff_utils import make_beta_schedule, default, extract_into_tensor, noise_like
from vmpro.models.ddim_sampler import DDIMSampler
from vmpro.models.layers.diff_LCN import get_lcn
from vmpro.models.layers.HRnet import _make_cls_head

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias!=None:
            m.bias.data.fill_(0.01)
    elif isinstance(m,nn.MultiheadAttention):
        nn.init.kaiming_normal_(m.in_proj_weight)
        nn.init.kaiming_normal_(m.out_proj.weight)
        m.in_proj_bias.data.fill_(0.01)
        m.out_proj.bias.data.fill_(0.01)


def clip_by_norm(layer, norm=1):
    if isinstance(layer, nn.Linear):
        if layer.weight.data.norm(2) > norm:
            layer.weight.data.mul_(norm / layer.weight.data.norm(2).item())



class Diff3DMesh(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, 
                    mesh_num_joints=None, 
                    flip_pairs=None,
                    vm_A=None,
                    selected_indices=None,
                    neighbour_matrix=None):
        super(Diff3DMesh, self).__init__()

        self.joint_num = cfg.dataset.num_joints
        self.actual_joint_num = self.joint_num
        self.depth_dim = cfg.model.simple3dpose.extra_depth_dim
        self.height_dim = cfg.model.heatmap_shape[0]
        self.width_dim = cfg.model.heatmap_shape[1]

        self.simple3dpose = simple3dpose.get_model(flip_pairs=flip_pairs)
        if cfg.model.simple3dpose.fix_network:
            for param in self.simple3dpose.parameters():
                param.requires_grad = False

        self.adaptive_A = nn.Sequential(
            nn.Linear(self.joint_num, mesh_num_joints*self.joint_num, bias = True),
            )
        if cfg.model.simple3dmesh.fix_network:
            for param in self.adaptive_A.parameters():
                param.requires_grad = False

        if cfg.model.simple3dmesh.fix_A:
            self.selected_indices = selected_indices
            self.vm_A = torch.tensor(vm_A).float().unsqueeze(0)

        self.model_diff = get_lcn(neighbour_matrix)
        if osp.isfile(cfg.model_diff.pretrained):
            pretrained = cfg.model_diff.pretrained
            pretrained_weight_dict = torch.load(pretrained,map_location='cpu')
            logging.info('=> loading pretrained model_diff from {}'.format(cfg.model_diff.pretrained))
            if 'model_state_dict' in pretrained_weight_dict:
                pretrained_state_dict = pretrained_weight_dict['model_state_dict']
            else:
                pretrained_state_dict = pretrained_weight_dict
            for key in list(pretrained_state_dict.keys()):
                if 'model_diff.' in key:
                    pretrained_state_dict[key[11:]] = pretrained_state_dict.pop(key)
            try:
                self.model_diff.load_state_dict(pretrained_state_dict, strict=True)
                logging.info(colored('Successfully load pretrained simple3dmesh model.', 'green'))
            except:
                try:
                    self.model_diff.load_state_dict(pretrained_state_dict, strict=False)
                    logging.info(colored('Load part of pretrained simple3dmesh model {} (strict=False)'.format(pretrained), 'green'))
                except:
                    logging.info(colored('Failed load pretrained simple3dmesh model {}'.format(pretrained), 'red'))
        else:
            logging.info('=> init model_diff weights from kaiming normal distribution')
            self.model_diff.apply(init_weights)
            self.model_diff.apply(clip_by_norm)
        
        if cfg.model_diff.fix_network:
            for param in self.model_diff.parameters():
                param.requires_grad = False

        # for heatmap2d
        if cfg.model.hrnet.heatmap2d:
            self.final_layer = nn.Conv2d(
                in_channels=48,
                out_channels=cfg.dataset.num_joints,
                kernel_size=cfg.model.hrnet.final_conv_kernel,
                stride=1,
                padding=1 if cfg.model.hrnet.final_conv_kernel == 3 else 0)

        # Diffusion parameters
        self.parameterization = cfg.diffusion.parameterization
        assert self.parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.register_schedule()
        self.ddim_steps = cfg.diffusion.ddim_steps
        self.clip_denoised = cfg.diffusion.clip_denoised
        self.log_every_t = cfg.diffusion.log_every_t
        self.ddim_sampler = DDIMSampler(self,concat=cfg.diffusion.uvd_feat,joint_num=self.joint_num)
        self.return_intermediates = cfg.diffusion.return_intermediates
        
        # if finetune_x0, we need a new adaptiveA layer
        if cfg.diffusion.finetune_x0 and not cfg.diffusion.share_adaptive_A:
            self.adaptive_A_ft = nn.Sequential(
                nn.Linear(self.joint_num, mesh_num_joints*self.joint_num, bias = True),
                )

    def register_schedule(self):
        betas = make_beta_schedule(
            schedule=cfg.diffusion.beta_schedule,
            n_timestep=cfg.diffusion.timesteps,
            linear_start=cfg.diffusion.linear_start, 
            linear_end=cfg.diffusion.linear_end,
            cosine_s=cfg.diffusion.cosine_s
        )

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = cfg.diffusion.linear_start
        self.linear_end = cfg.diffusion.linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.v_posterior = 0
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas', to_torch(np.sqrt(1. / alphas)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        model_out = self.model_diff(x_noisy, t, condition=cond)

        if isinstance(model_out, tuple) and not return_ids:
            return model_out[0]
        else:
            return model_out

    def q_sample(self, x_start, time_t, noise):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, time_t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, time_t, x_start.shape) * noise)

    @torch.no_grad()
    def sample(self, cond, batch_size, given_x_t=None):
        shape = (batch_size, self.actual_joint_num * 3)
        denoise_uvd, intermediates = self.p_sample_loop(shape, cond, given_x_t)
        return denoise_uvd, intermediates

    @torch.no_grad()
    def sample_ddim(self, cond, batch_size, **kwargs):
        shape = (batch_size, self.actual_joint_num * 3)
        if cfg.diffusion.uvd_feat:
            self.device = cond['concat'].device
        elif cfg.diffusion.global_feat:
            self.device = cond['guide']['global'][0].device
        elif cfg.diffusion.local_feat:
            self.device = cond['guide']['local'][0].device
        self.ddim_sampler.device = self.device
        denoise_uvd, intermediates = self.ddim_sampler.sample(self.ddim_steps, shape, cond, verbose=False, **kwargs)
        return denoise_uvd, intermediates

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond=None, clip_denoised=True):
        model_out = self.model_diff(x, t, condition=cond)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1*cfg.diffusion.data_scale, cfg.diffusion.data_scale)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond=None, clip_denoised=True, repeat_noise=False):
        batch_size, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x).to(device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None, given_x_t=None):
        device = self.betas.device
        if cfg.diffusion.infer_start == 'z0':
            denoise_uvd = torch.zeros(shape, device=device)
        elif cfg.diffusion.infer_start == 'rand':
            denoise_uvd = torch.randn(shape, device=device)
        elif given_x_t is not None:
            denoise_uvd = given_x_t        # (1, 3, H, W)
        if cfg.diffusion.cond_content == '3d+img':
            concat = cond['concat'].view(denoise_uvd.view(shape[0],self.actual_joint_num,-1))
            cond = cond['guide']
        intermediates = []
        # for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
        for i in tqdm(reversed(range(0, self.num_timesteps))):
            if cfg.diffusion.cond_content == '3d+img':
                denoise_uvd = torch.cat([denoise_uvd.view(shape[0],self.actual_joint_num,-1),concat],dim=-1).view(shape[0],self.actual_joint_num*6)
            denoise_uvd = self.p_sample(denoise_uvd, torch.full((shape[0],), i, device=device, dtype=torch.long), cond=cond, clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(denoise_uvd)
        return denoise_uvd, intermediates

    def forward(self, x, trans_inv, intrinsic_param, joint_root, depth_factor, gt_uvd_jts=None, flip_item=None, flip_output=False, flip_mask=None, is_train=True):
        batch_size = joint_root.shape[0]
        device = joint_root.device

        if gt_uvd_jts is not None:
            gt_uvd_jts[..., :2] = gt_uvd_jts[..., :2] + 0.5*(1-cfg.model_diff.zero_center)
            gt_uvd_jts = gt_uvd_jts * cfg.diffusion.data_scale
            gt_uvd_jts = gt_uvd_jts.reshape((batch_size, self.actual_joint_num*3))

        pred_xyz_jts_ret, pred_uvd_jts_flat, adaptive_A, confidence_ret, mesh3d, pred_uv_jts_flat, heatmaps_2d = None, None, None, None, None, None, None
        
        output = self.simple3dpose(x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item, flip_output, flip_mask, return_feat=True, is_train=is_train)       # (B, J+K, 3), (B, J+K)

        # ######################## VMarker mesh estimates ########################
        pred_xyz_jts, confidence, pred_uvd_jts_flat, pred_root_xy_img = output['pred_xyz_jts'], output['confidence'], output['pred_uvd_jts_flat'], output['pred_root_xy_img']       # (B, J+K, 3), (B, J+K)
        features = output['features']   # B, C, H, W

        confidence_ret = confidence.clone()
        pred_xyz_jts_ret = pred_xyz_jts.clone()
    
        # detach pose3d to mesh for faster convergence
        pred_xyz_jts = pred_xyz_jts.detach()
        confidence = confidence.detach()

        # get adaptive_A based on the estimation confidence 
        adaptive_A = self.adaptive_A(confidence.view(confidence.shape[0], -1))
        adaptive_A = adaptive_A.view(adaptive_A.size(0), -1, self.joint_num)   # B, V, J+K

        # get mesh by production of 3D pose & reconstruction matrix A
        mesh3d = torch.matmul(adaptive_A, pred_xyz_jts)     # B, V, 3
        

        # ######################## Prepare condition for VMarker-Pro ########################
        # features B, C, H, W
        # grid B, 1, J+K, 2
        cond_local, cond_global, cond_uvd = None, None, None
        # local_feat
        if cfg.diffusion.local_feat:
            pred_uvd_jts = pred_uvd_jts_flat.reshape((batch_size, self.actual_joint_num, 3))
            #  -1 ~ 1
            coord_x, coord_y = pred_uvd_jts[...,:1]*2, pred_uvd_jts[...,1:2]*2
            pred_uv_jts_coord = torch.cat((coord_x, coord_y), dim=2).clone().unsqueeze(1)     # B, J+K, 2 -> B, 1, J+K, 2
            pred_uv_jts_coord = torch.clamp(pred_uv_jts_coord, -1, 1)

            cond_local = F.grid_sample(features[0], pred_uv_jts_coord, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2).permute(0, 2, 1)  # B, C, J+K -> B, J+K, C
        # global_feat
        if cfg.diffusion.global_feat:
            cond_global = features
        # uvd_feat
        if cfg.diffusion.uvd_feat:
            pred_uvd_jts_new = pred_uvd_jts_flat.clone()
            pred_uvd_jts_new[..., :2] = pred_uvd_jts_new[..., :2] + 0.5*(1-cfg.model_diff.zero_center)
            pred_uvd_jts_new = pred_uvd_jts_new * cfg.diffusion.data_scale
            pred_uvd_jts_new = pred_uvd_jts_new.reshape((batch_size, self.actual_joint_num,3))
            if cfg.diffusion.uvd_dim == '3d':
                cond_uvd = pred_uvd_jts_new
            elif cfg.diffusion.uvd_dim == '2d':
                cond_uvd = pred_uvd_jts_new[...,:2]


        # ######################## Train & inference VMarker-Pro ########################
        if is_train:
            # sample a timestep 
            time_t = torch.randint(0, self.num_timesteps, size=(batch_size // 2 + 1,)).to(device).long()
            time_t = torch.cat([time_t, self.num_timesteps - time_t - 1], dim=0)[:batch_size]

            if cfg.diffusion.debug_timestep != -1:
                time_t[:] = cfg.diffusion.debug_timestep

            noise = torch.randn_like(gt_uvd_jts)   # (B, J*3)  
            x_noisy = self.q_sample(x_start=gt_uvd_jts, time_t=time_t, noise=noise)        # (1, 3, H, W)
            if cfg.diffusion.uvd_feat:
                x_in = torch.cat([x_noisy.view(batch_size,self.joint_num,3),cond_uvd.clone()],dim=-1)
                # uvd_dim ('2d') [batch_size, self.actual_joint_num*5]
                # uvd_dim ('3d') [batch_size, self.actual_joint_num*6]
                x_in = x_in.view(batch_size, -1)
            else:
                x_in = x_noisy
            cond_feat_in = {"global":cond_global,"local":cond_local}
            model_out = self.model_diff(x_in, time_t, condition=cond_feat_in.copy())

            if self.parameterization == "eps":
                target = noise
                x_recon = (x_noisy - model_out.clone()*extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, time_t, x_noisy.shape))/extract_into_tensor(self.sqrt_alphas_cumprod, time_t, x_noisy.shape)
            elif self.parameterization == "x0":
                target = gt_uvd_jts
                x_recon = model_out.clone()
            else:
                raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")
            x_recon = x_recon.reshape(batch_size, -1, 3)
            x_recon = x_recon / cfg.diffusion.data_scale
            x_recon[..., :2] = x_recon[..., :2] - 0.5*(1-cfg.model_diff.zero_center)

            
            # ######################## Regress heatmap2d ########################
            if cfg.model.hrnet.heatmap2d:
                heatmaps_2d = self.final_layer(features[0])     # B, J+K, H, W

            return pred_xyz_jts_ret, pred_uvd_jts_flat, adaptive_A, confidence_ret, mesh3d, None, heatmaps_2d, model_out, target, x_recon.reshape(batch_size, -1, 3), pred_uv_jts_flat
        else:
            with torch.no_grad():
                denoise_uvd = None
                cond_feat = {'guide':{"global":cond_global,"local":cond_local},'concat':cond_uvd}
                if self.ddim_steps:
                    denoise_uvd, intermediates = self.sample_ddim(cond_feat, batch_size, given_x_t=denoise_uvd)
                else:
                    denoise_uvd, intermediates = self.sample(cond_feat, batch_size, given_x_t=denoise_uvd)

                denoise_uvd = denoise_uvd.reshape(batch_size, -1, 3)
                denoise_uvd = denoise_uvd / cfg.diffusion.data_scale
                denoise_uvd[..., :2] = denoise_uvd[..., :2] - 0.5*(1-cfg.model_diff.zero_center)

                x_recon = denoise_uvd.clone()
                pred_uvd_jts = denoise_uvd.clone()
                pred_uvd_jts = pred_uvd_jts.reshape((batch_size, self.actual_joint_num, 3))
                
                # ######### use denoised uvd to get mesh3d #########
                heatmaps = output['heatmaps']   # B, J+K, D, H, W

                coord_x, coord_y, coord_z = pred_uvd_jts[...,:1], pred_uvd_jts[...,1:2], pred_uvd_jts[...,2:]

                coord_x = (coord_x+0.5) * float(self.width_dim)
                coord_y = (coord_y+0.5) * float(self.height_dim)
                coord_z = (coord_z+0.5) * float(self.depth_dim)
                coord_x = torch.clamp(coord_x, 0, self.width_dim-1)
                coord_y = torch.clamp(coord_y, 0, self.height_dim-1)
                coord_z = torch.clamp(coord_z, 0, self.depth_dim-1)
                pred_uvd_jts_coord = torch.cat((coord_x, coord_y, coord_z), dim=2).clone()     # B, J+K, 3

                # NOTE that heatmap is (z, y, x) pred_uvd_jts is (x, y, z)
                pred_uvd_jts_ind = (pred_uvd_jts_coord[...,2]*self.depth_dim*self.height_dim + pred_uvd_jts_coord[...,1]*self.height_dim + pred_uvd_jts_coord[...,0]).unsqueeze(2).long()
                heatmaps = heatmaps.reshape(batch_size, self.actual_joint_num, -1)   # B*(J+K), 1, D, H, W -> B, J+K, D*H*W
                confidence = torch.gather(heatmaps, 2, pred_uvd_jts_ind).squeeze(-1)      # B, J+K
                
                del heatmaps

                output = self.simple3dpose(torch.zeros(1).float(), trans_inv, intrinsic_param, joint_root, depth_factor, None, False, None, given_uvd_jts=pred_uvd_jts, given_confidence=confidence)
                
                pred_xyz_jts = output['pred_xyz_jts']   # B, J+K, 3
                pred_xyz_jts_ret = pred_xyz_jts.clone()
                confidence_ret = output['confidence']
                pred_root_xy_img = output['pred_root_xy_img']

                adaptive_A = self.adaptive_A(confidence.view(confidence.shape[0], -1))
                adaptive_A = adaptive_A.view(adaptive_A.size(0), -1, self.joint_num)   # B, V, J+K

                # get mesh by production of 3D pose & reconstruction matrix A
                mesh3d = torch.matmul(adaptive_A, pred_xyz_jts)     # B, V, 3
                # ######### use denoised uvd to get mesh3d #########

                if cfg.model.simple3dmesh.fix_A:
                    mesh3d[:, self.selected_indices] = torch.matmul(self.vm_A.transpose(1,2).to(device=adaptive_A.device), pred_xyz_jts[:, 17:])

                inter_mesh3d_list = []

                return pred_xyz_jts_ret, pred_uvd_jts, adaptive_A, confidence_ret, mesh3d, inter_mesh3d_list, None, pred_root_xy_img, x_recon


def get_model(mesh_num_joints, flip_pairs, vm_A=None, selected_indices=None, neighbour_matrix=None):
    model = Diff3DMesh(mesh_num_joints=mesh_num_joints, flip_pairs=flip_pairs, vm_A=vm_A, selected_indices=selected_indices, neighbour_matrix=neighbour_matrix)
    if osp.isfile(cfg.diffusion.pretrained):
        pretrained = cfg.diffusion.pretrained
        logging.info('==> try loading pretrained diff3dmesh model {}'.format(pretrained))
        pretrained_weight_dict = torch.load(pretrained,map_location='cpu')
        if 'model_state_dict' in pretrained_weight_dict:
            pretrained_state_dict = pretrained_weight_dict['model_state_dict']
        else:
            pretrained_state_dict = pretrained_weight_dict
        try:
            model.load_state_dict(pretrained_state_dict, strict=True)
            logging.info(colored('Successfully load pretrained diff3dmesh model.', 'green'))
        except:
            try:
                model.load_state_dict(pretrained_state_dict, strict=False)
                logging.info(colored('Load part of pretrained diff3dmesh model {} (strict=False)'.format(pretrained), 'green'))
            except:
                logging.info(colored('Failed load pretrained diff3dmesh model {}'.format(pretrained), 'red'))
    if osp.isfile(cfg.model.simple3dmesh.pretrained):
        pretrained = cfg.model.simple3dmesh.pretrained
        logging.info('==> try loading pretrained simple3dmesh model {}'.format(pretrained))
        pretrained_weight_dict = torch.load(pretrained,map_location='cpu')
        if 'model_state_dict' in pretrained_weight_dict:
            pretrained_state_dict = pretrained_weight_dict['model_state_dict']
        else:
            pretrained_state_dict = pretrained_weight_dict
        for key in list(pretrained_state_dict.keys()):
            if 'simple3dpose' not in key and 'adaptive_A' not in key:
                pretrained_state_dict['simple3dpose.' + key] = pretrained_state_dict.pop(key)
        try:
            model.load_state_dict(pretrained_state_dict, strict=True)
            logging.info(colored('Successfully load pretrained simple3dmesh model.', 'green'))
        except:
            try:
                model.load_state_dict(pretrained_state_dict, strict=False)
                logging.info(colored('Load part of pretrained simple3dmesh model {} (strict=False)'.format(pretrained), 'green'))
            except:
                logging.info(colored('Failed load pretrained simple3dmesh model {}'.format(pretrained), 'red'))
    return model