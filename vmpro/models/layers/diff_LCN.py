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
from vmpro.models.layers.HRnet import _make_cls_head


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    from https://github.com/wzlxjtu/PositionalEncoding2D
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class LCN(nn.Module):
    def __init__(self, neighbour_matrix=None):
        super(LCN, self).__init__()
        self.inplane = cfg.model_diff.inplane
        self.joint_num = cfg.dataset.num_joints
        self.num_blocks = cfg.model_diff.num_blocks
        self.dropout_rate = cfg.model_diff.dropout_rate
        self.heads = cfg.model_diff.attn_head
        self.cond_content = cfg.diffusion.cond_content
        self.concat = cfg.diffusion.uvd_feat
        self.concat_dim = 3*self.concat if cfg.diffusion.uvd_dim == '3d' else 2*self.concat
        self.global_feat = cfg.diffusion.global_feat
        self.local_feat = cfg.diffusion.local_feat
        self.agg_feat = cfg.diffusion.agg_feat
        self.avg_feat = cfg.diffusion.avg_feat
        assert neighbour_matrix.shape[0] == neighbour_matrix.shape[1]
        assert neighbour_matrix.shape[0] == self.joint_num
        self.neighbour_matrix = neighbour_matrix
        self.init_mask()  # init self.mask
        
        # condition type
        self.cond_method = cfg.diffusion.cond_method
        # first layer
        self.linear_start = nn.Linear(self.joint_num*(3+self.concat_dim), self.joint_num*self.inplane)
        if cfg.model_diff.bn_norm == 'batch':
            self.bn_start = nn.BatchNorm1d(self.inplane)
        elif cfg.model_diff.bn_norm == 'group':
            self.bn_start = nn.GroupNorm(32, self.inplane)
        if cfg.model_diff.act_func == 'LeakyReLU':
            self.activation_start = nn.LeakyReLU(negative_slope=0.2)
        elif cfg.model_diff.act_func == 'SiLU':
            self.activation_start = nn.SiLU()
        self.dropout_start = nn.Dropout(p=self.dropout_rate)

        # final layer
        self.linear_final = nn.Linear(self.joint_num*self.inplane, self.joint_num*3)

        # blocks
        self.linear_blocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.joint_num*self.inplane, self.joint_num*self.inplane),
                          nn.Linear(self.joint_num*self.inplane, self.joint_num*self.inplane)]) for _ in range(self.num_blocks)])

        self.bn_blocks = nn.ModuleList([
            nn.ModuleList([self.bn_start, 
                           self.bn_start]) for _ in range(self.num_blocks)])

        self.rest_blocks = nn.ModuleList([
            nn.ModuleList([nn.Sequential(
                self.activation_start,
                self.dropout_start),
                           nn.Sequential(
                self.activation_start,
                self.dropout_start)]) for _ in range(self.num_blocks)])

        # condition blocks (concat)
        if cfg.diffusion.global_feat:
            self.cond_linear = nn.Sequential(
                nn.Linear(cfg.model_diff.condplane, self.inplane),
                self.activation_start)
        if cfg.diffusion.global_feat:
            self.feat_size= cfg.model_diff.feat_size
            self.cond_num = cfg.model_diff.feat_size**2
        else:
            self.cond_num = self.joint_num
        self.cond_blocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(2*self.joint_num*self.inplane, self.joint_num*self.inplane)]) for _ in range(self.num_blocks)])   
        
        if self.local_feat:
            self.cond_layer_local = nn.Sequential(
                nn.Linear(48, self.inplane),
                self.activation_start) 
        
        if self.global_feat:
            if self.avg_feat:
                self.avg_pool = nn.AdaptiveAvgPool2d((cfg.model_diff.feat_size,cfg.model_diff.feat_size))
            if self.agg_feat:
                self.incre_modules, self.downsamp_modules, self.final_feat_layer = _make_cls_head()
            if self.cond_method == 'att':
                self.multiatten = nn.ModuleList([
                    nn.ModuleList([nn.MultiheadAttention(embed_dim=self.inplane, num_heads=self.heads,batch_first = True),
                                nn.MultiheadAttention(embed_dim=self.inplane, num_heads=self.heads,batch_first = True)])for _ in range(self.num_blocks)])

        # time embedding dimension
        self.time_ch = cfg.diffusion.time_ch 
        self.time_emb_dim = self.time_ch*4
        self.temb_dense = nn.ModuleList([
            nn.Linear(self.time_ch, self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.inplane),])

    
    def init_mask(self):
        """
        Only support locally_connected
        """
        assert self.neighbour_matrix is not None
        L = self.neighbour_matrix.T
        assert L.shape == (self.joint_num, self.joint_num)
        self.mask = torch.from_numpy(L)

    def mask_weights(self, layer):
        assert isinstance(layer, nn.Linear), 'masked layer must be linear layer'

        output_size, input_size = layer.weight.shape  # pytorch weights [output_channel, input_channel]
        input_size, output_size = int(input_size), int(output_size)
        assert input_size % self.joint_num == 0 and output_size % self.joint_num == 0
        in_F = int(input_size / self.joint_num)
        out_F = int(output_size / self.joint_num)
        weights = layer.weight.data.view([self.joint_num, out_F, self.joint_num, in_F])
        weights.mul_(self.mask.t().view(self.joint_num, 1, self.joint_num, 1).to(device=weights.get_device()))

    def forward(self, xin, time_t, condition=None):
        """
        Param:
            xin: [N, J*3]
            time_t: [N]
            cond: [N, J, D=1, H, W]
        """
        batch_size = xin.shape[0]
        # #### mask weights of all linear layers before forward
        self.mask_weights(self.linear_start)
        for block_idx in range(self.num_blocks):
            for layer_idx in range(2):
                self.mask_weights(self.linear_blocks[block_idx][layer_idx])
        self.mask_weights(self.linear_final)

        # #### time embedding
        temb = get_timestep_embedding(time_t, self.time_ch)
        temb = self.temb_dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb_dense[1](temb)
        temb = nonlinearity(temb)
        temb = self.temb_dense[2](temb)
        temb = nonlinearity(temb)
        temb = temb.unsqueeze(1).repeat(1, self.joint_num, 1)   # [N, J, C]
        temb = temb.view(-1, self.joint_num*self.inplane)  # [N, J*C]

        # #### first layer
        x = self.linear_start(xin)
        # #### add time embedding
        x += temb   # [N, J*C]
        
        # #### condition
        if self.global_feat:
            if self.agg_feat:
                y = self.incre_modules[0](condition['global'][0])
                for i in range(len(self.downsamp_modules)):
                    y = self.incre_modules[i + 1](condition['global'][i + 1]) + \
                        self.downsamp_modules[i](y)
                cond_feat = self.final_feat_layer(y)
            else:
                cond_feat = condition['global'][0]
            if cfg.diffusion.avg_feat:
                cond_feat = self.avg_pool(cond_feat)
            cond_feat = torch.flatten(cond_feat,start_dim=2, end_dim=3).transpose(2,1)
            cond = self.cond_linear(cond_feat)

        if self.local_feat:
            cond_local = self.cond_layer_local(condition['local']).contiguous().view(-1, self.joint_num*self.inplane)
            x += cond_local  # [N, J*C]

        x = x.view(-1, self.joint_num, self.inplane).transpose(2, 1)  # [N, C, J]
        x = self.bn_start(x)    # [N, C, J]
        x = x.transpose(2, 1).contiguous().view(-1, self.joint_num*self.inplane)  # [N, J*C]
        x = self.activation_start(x)
        x = self.dropout_start(x)   # [N, J*C]

        if self.cond_method == 'att' and self.global_feat:
            emb_x = get_timestep_embedding(torch.arange(0,self.joint_num),self.inplane).cuda().view(1,self.joint_num,self.inplane).cuda()
            emb_ctx = positionalencoding2d(self.inplane, self.feat_size, self.feat_size).unsqueeze(dim=0).cuda()
            emb_ctx = torch.flatten(emb_ctx,start_dim=2, end_dim=3).transpose(2,1)

        for block_idx in range(self.num_blocks):
            x_res = x.clone()
            for layer_idx in range(2):
                x = self.linear_blocks[block_idx][layer_idx](x)

                # import ipdb; ipdb.set_trace()
                if self.global_feat:
                    if self.cond_method == 'att':
                        cond = cond.contiguous().view(-1, self.cond_num, self.inplane)
                        x0 = x.contiguous().view(-1, self.joint_num, self.inplane)
                        x0 = x0 + emb_x
                        context = self.multiatten[block_idx][layer_idx](query = x0,key = cond+emb_ctx,value = cond)[0]
                        context = context.contiguous().view(-1, self.joint_num*self.inplane)
                    else:
                        context = cond.contiguous().view(-1, self.joint_num*self.inplane)
                    x += temb   # [N, J*C]
                    x += context
                else:
                    x += temb

                x = x.view((-1, self.joint_num, self.inplane)).transpose(2, 1)     # [N, C, J]
                x = self.bn_blocks[block_idx][layer_idx](x)     # [N, C, J]
                x = x.transpose(2, 1).contiguous().view(-1, self.joint_num*self.inplane)        # [N, J*C]
                x = self.rest_blocks[block_idx][layer_idx](x)
            x = x_res + x
        # #### final layer
        x = self.linear_final(x)  # [N, J*3]
        x = x.view(-1, self.joint_num, 3)  # [N, J, 3]
        xin = xin.view(-1, self.joint_num, 3+self.concat_dim)  # [N, J, 3]
        x = xin[:,:,:3] + x
        x = x.view(-1, self.joint_num*3)  # [N, J*3]
        return x

def get_lcn(neighbour_matrix):
    model = LCN(neighbour_matrix)
    return model