import os.path as osp
import numpy as np
import cv2, os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
import pickle
from itertools import islice
import json
import time
import scipy.sparse as ssp
import matplotlib.pyplot as plt
import logging

from vmpro.core.base import prepare_network
from vmpro.core.config import cfg
from vmpro.utils.funcs_utils import lr_check, save_obj
from vmpro.utils.vis import vis_joints_3d, render_mesh, denormalize_image, vis_heatmap_2d
from vmpro.utils.aug_utils import get_affine_transform, flip_img, augm_params


class Simple3DMeshTrainer:
    def __init__(self, args, load_path, writer=None, master=None):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history,\
            self.dataset, self.sampler,__ = prepare_network(args, load_path=load_path, is_train=True, master=master)

        self.main_dataset = self.dataset_list[0]
        self.joint_num = cfg.dataset.num_joints
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.print_freq = cfg.train.print_freq
        self.vis_freq = cfg.train.vis_freq
        self.writer = writer
        self.device = args.device

        self.J_regressor = torch.Tensor(self.main_dataset.joint_regressor).cuda()
        self.selected_indices = self.main_dataset.selected_indices
        self.vm_B = torch.Tensor(self.main_dataset.vm_B).cuda()
        self.edge_add_epoch = cfg.train.edge_loss_start

    def train(self, epoch, n_iters_total, master, ema_helper=None):
        self.model.train()
        
        metric_dict = defaultdict(list)

        lr_check(self.optimizer, epoch, master)

        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator, dynamic_ncols=True) if master else self.batch_generator
        for i, meta in enumerate(batch_generator):
            for k, _ in meta.items():
                meta[k] = meta[k].cuda()
                if k == 'img':
                    meta[k] = meta[k].requires_grad_()

            imgs = meta['img'].cuda()
            inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
            gt_pose_root = meta['root_cam'].cuda()
            depth_factor = meta['depth_factor'].cuda()
            flip_mask = meta['flip'].cuda().reshape(-1) if cfg.aug.flip else None
            batch_size = imgs.shape[0]

            joint_uvd_valid, joint_cam_valid, mesh_valid = meta['joint_uvd_valid'].cuda(), meta['joint_cam_valid'].cuda(), meta['mesh_valid'].cuda()
            gt_pose, gt_uvd_pose, gt_mesh = meta['joint_cam'].cuda(), meta['joint_uvd'].cuda(), meta['mesh_cam'].cuda()
            
            _, pred_uvd_pose, _, confidence, pred_mesh, _, _ = self.model(imgs, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_item=None, flip_mask=flip_mask)
            pred_pose = torch.cat((torch.matmul(self.J_regressor, pred_mesh), torch.matmul(self.vm_B.T[None], pred_mesh[:, self.selected_indices])), dim=1)

            joint3d_loss = self.loss['joint_3d'](pred_uvd_pose, gt_uvd_pose, joint_uvd_valid)
            joint3d_reg_loss = self.loss['joint_reg_3d'](pred_pose, gt_pose, joint_cam_valid)
            conf_loss = self.loss['conf'](confidence, joint_uvd_valid)
            mesh3d_loss = self.loss['mesh_3d'](pred_mesh, gt_mesh, mesh_valid)
            mesh3d_normal_loss = self.loss['normal'](pred_mesh, gt_mesh)
            mesh3d_edge_loss = self.loss['edge'](pred_mesh, gt_mesh)

            loss = cfg.loss.loss_weight_joint3d * joint3d_loss + \
                cfg.loss.loss_weight_joint3d_reg * joint3d_reg_loss + \
                cfg.loss.loss_weight_conf * conf_loss + \
                cfg.loss.loss_weight_mesh3d * mesh3d_loss + \
                cfg.loss.loss_weight_normal * mesh3d_normal_loss
            
            loss = loss*cfg.loss.loss_weight

            metric_dict['joint3d_loss'].append(joint3d_loss.item())
            metric_dict['joint3d_reg_loss'].append(joint3d_reg_loss.item())
            metric_dict['conf_loss'].append(conf_loss.item())
            metric_dict['mesh3d_loss'].append(mesh3d_loss.item())
            metric_dict['mesh3d_normal_loss'].append(mesh3d_normal_loss.item())
            metric_dict['mesh3d_edge_loss'].append(mesh3d_edge_loss.item())
            
            if epoch > self.edge_add_epoch:
                loss += cfg.loss.loss_weight_edge * mesh3d_edge_loss
            metric_dict['total_loss'].append(loss.item())

            self.optimizer['3d'].zero_grad()
            self.optimizer['mesh'].zero_grad()
            loss.backward()
            self.optimizer['3d'].step()
            self.optimizer['mesh'].step()

            running_loss += float(loss.detach().item())

            if master:
                if i % self.print_freq == 0:
                    batch_generator.set_description(f'Epoch{epoch}_({i}/{len(self.batch_generator)}) => '
                                                    f'3d: {joint3d_loss.detach():.4f}, '
                                                    f'3d_r: {joint3d_reg_loss.detach():.4f}, '
                                                    f'conf: {conf_loss.detach():.4f}, '
                                                    f'mesh: {mesh3d_loss.detach():.4f}, '
                                                    f'norm: {mesh3d_normal_loss.detach():.4f}, '
                                                    f'edge: {mesh3d_edge_loss.detach():.4f}, '
                                                    f'total: {loss.detach():.4f} ')
                if i % self.vis_freq == 0:
                    nrow = min(max(batch_size//3, 1), 4)
                    ncol = (min(batch_size//nrow, 3))*2
                    vis_joints_3d(imgs.detach().cpu().numpy(), gt_pose.detach().cpu().numpy(), None, file_name='train_{:08}_joint3d_gt.jpg'.format(i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                    vis_joints_3d(imgs.detach().cpu().numpy(), pred_pose.detach().cpu().numpy(), None, file_name='train_{:08}_joint3d_pred.jpg'.format(i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                    vis_joints_3d(imgs.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy(), None, file_name='train_{:08}_mesh_gt.jpg'.format(i), nrow=nrow, ncol=ncol)
                    vis_joints_3d(imgs.detach().cpu().numpy(), pred_mesh.detach().cpu().numpy(), None, file_name='train_{:08}_mesh_pred.jpg'.format(i), nrow=nrow, ncol=ncol)


                for title, value in metric_dict.items():
                    self.writer.add_scalar("{}/{}".format('train', title), value[-1], n_iters_total)
                n_iters_total += 1

        self.loss_history.append(running_loss / len(self.batch_generator))

        logging.info(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')
        return n_iters_total

class Simple3DMeshTester:
    def __init__(self, args, load_path='', writer=None, master=None):
        self.val_loader_list, self.val_dataset, self.model,_, _, _, _, _, _, _, _ = \
            prepare_network(args, load_path=load_path, is_train=False, master=master)

        self.joint_num = self.val_dataset[0].joint_num
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.print_freq = cfg.train.print_freq
        self.vis_freq = cfg.test.vis_freq
        self.writer = writer
        self.device = args.device

        self.J_regressor = torch.Tensor(self.val_dataset[0].joint_regressor).cuda()
        self.selected_indices = self.val_dataset[0].selected_indices
        self.vm_B = torch.Tensor(self.val_dataset[0].vm_B).cuda()

    def test(self, epoch, master, world_size, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        for dataset_name, val_dataset, val_loader in zip(cfg.dataset.test_list, self.val_dataset, self.val_loader_list):
            results = defaultdict(list)
            metric_dict = defaultdict(list)

            joint_error = 0.0
            logging.info(f'=> Evaluating on {dataset_name} ...')
            loader = tqdm(val_loader, dynamic_ncols=True) if master else val_loader
            with torch.no_grad():
                for i, meta in enumerate(loader):
                    for k, _ in meta.items():
                        meta[k] = meta[k].cuda()

                    imgs = meta['img'].cuda()
                    batch_size = imgs.shape[0]
                    inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
                    depth_factor, gt_pose_root = meta['depth_factor'].cuda(), meta['root_cam'].cuda()

                    gt_pose, gt_mesh = meta['joint_cam'].cuda(), meta['mesh_cam'].cuda()

                    _, pred_uvd_pose, _, confidence, pred_mesh, _, _ = self.model(imgs, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_mask=None, is_train=False)

                    results['gt_pose'].append(gt_pose.detach().cpu().numpy())
                    results['gt_mesh'].append(gt_mesh.detach().cpu().numpy())

                    # flip_test
                    if isinstance(imgs, list):
                        imgs_flip = [flip_img(img.clone()) for img in imgs]
                    else:
                        imgs_flip = flip_img(imgs.clone())

                    pred_vm_flip, _, _, _, pred_mesh_flip, _, _ = self.model(imgs_flip, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_item=(pred_uvd_pose, confidence), flip_output=True, flip_mask=None, is_train=False)

                    pred_pose_flip = torch.cat((torch.matmul(self.J_regressor, pred_mesh_flip), torch.matmul(self.vm_B.T[None], pred_mesh_flip[:, self.selected_indices])), dim=1)

                    results['pred_vm_flip'].append(pred_vm_flip.detach().cpu().numpy())
                    results['pred_pose_flip'].append(pred_pose_flip.detach().cpu().numpy())
                    results['pred_mesh_flip'].append(pred_mesh_flip.detach().cpu().numpy())
                    results['gt_pose_root'].append(gt_pose_root.detach().cpu().numpy())
                    results['focal_l'].append(meta['focal_l'].detach().cpu().numpy())
                    results['center_pt'].append(meta['center_pt'].detach().cpu().numpy())
                    results['idx'].append(meta['idx'].detach().cpu().numpy())

                    j_error = val_dataset.compute_joint_err(pred_pose_flip, gt_pose)

                    if master:
                        if i % self.print_freq == 0:
                            loader.set_description(f'{eval_prefix}({i}/{len(val_loader)}) => joint error: {j_error:.4f}')
                        if cfg.test.vis and i % self.vis_freq == 0:
                            nrow = min(max(batch_size//3, 1), 4)
                            ncol = (min(batch_size//nrow, 3))*2
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_pose.detach().cpu().numpy(), None, file_name='val_{}_{:08}_joint3d_gt.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_gt.jpg'.format(dataset_name, i), nrow=nrow, ncol=ncol)
                            vis_joints_3d(imgs.detach().cpu().numpy(), pred_pose_flip.detach().cpu().numpy(), None, file_name='val_{}_{:08}_joint3d_pred_flip.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                            vis_joints_3d(imgs.detach().cpu().numpy(), pred_mesh_flip.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_pred_flip.jpg'.format(dataset_name, i), nrow=nrow, ncol=ncol)

                        joint_error += j_error
                for term in results.keys():
                    results[term] = np.concatenate(results[term])
            
                self.joint_error = joint_error / max(len(val_loader),1)

                if master:
                    result_path = osp.join(cfg.metric_dir, "{}_result_valset.pkl".format(dataset_name))
                    with open(result_path, 'wb') as f:
                        pickle.dump({
                            'gt_mesh': results['gt_mesh'],
                            'pred_vm_flip': results['pred_vm_flip'],
                            'pred_mesh_flip': results['pred_mesh_flip'],
                            'pred_pose_flip': results['pred_pose_flip'],
                            'gt_pose': results['gt_pose'],
                        }, f)
                    joint_flip_error_dict = val_dataset.compute_per_joint_err_dict(results['pred_pose_flip'], results['gt_pose'])
                    mesh_flip_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh_flip'], results['gt_mesh'],\
                        results['pred_pose_flip'], results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name, is_flip=True)

                    msg = ''
                    msg += f'\n{eval_prefix}'
                    for metric_key in joint_flip_error_dict.keys():
                        metric_dict[metric_key+'_REG'].append(joint_flip_error_dict[metric_key].item())
                        msg += f' | {metric_key:12}: {joint_flip_error_dict[metric_key]:3.2f}'
                    msg += f'\n{eval_prefix}'
                    for metric_key in mesh_flip_error_dict.keys():
                        metric_dict[metric_key].append(mesh_flip_error_dict[metric_key].item()) 
                        msg += f' | {metric_key:12}: {mesh_flip_error_dict[metric_key]:3.2f}'
                    print(msg)

                    for title, value in metric_dict.items():
                        self.writer.add_scalar("{}_{}/{}_epoch".format('val', dataset_name, title), value[-1], epoch)
                        
                    result_path = osp.join(cfg.metric_dir, "{}_result_valset.pkl".format(dataset_name))
                    with open(result_path, 'wb') as f:
                        pickle.dump({
                            'idx': results['idx'],
                            'f':results['focal_l'],
                            'c':results['center_pt'],
                            'gt_mesh': results['gt_mesh'],
                            'pred_vm_flip': results['pred_vm_flip'],
                            'pred_mesh_flip': results['pred_mesh_flip'],
                            'pred_pose_flip': results['pred_pose_flip'],
                            'gt_pose': results['gt_pose'],
                        }, f)

                    # saving metric
                    metric_path = osp.join(cfg.metric_dir, "{}_metric_e{}_valset.json".format(dataset_name, epoch))
                    with open(metric_path, 'w') as fout:
                        json.dump(metric_dict, fout, indent=4, sort_keys=True)
                    print(f'=> writing metric dict to {metric_path}')


class Diffusion3DMeshTrainer:
    def __init__(self, args, load_path, writer=None, master=None):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history,\
            self.dataset, self.sampler,self.ema_helper = prepare_network(args, load_path=load_path, is_train=True, master=master)

        self.main_dataset = self.dataset_list[0]
        self.joint_num = cfg.dataset.num_joints
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.print_freq = cfg.train.print_freq
        self.vis_freq = cfg.train.vis_freq
        self.writer = writer
        self.device = args.device

        self.J_regressor = torch.Tensor(self.main_dataset.joint_regressor).cuda()
        self.selected_indices = self.main_dataset.selected_indices
        self.vm_B = torch.Tensor(self.main_dataset.vm_B).cuda()
        self.edge_add_epoch = cfg.train.edge_loss_start
    
        # Diffusion parameters
        self.parameterization = cfg.diffusion.parameterization
        assert self.parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'


    def train(self, epoch, n_iters_total, master):
        if cfg.model.simple3dpose.fix_network:
            if cfg.model.simple3dmesh.fix_network:
                try: 
                    self.model.module.simple3dpose.eval()
                    self.model.module.adaptive_A.eval()
                    self.model.module.model_diff.train()                
                except:
                    self.model.simple3dpose.eval()
                    self.model.adaptive_A.eval()
                    self.model.model_diff.train()  
            else:
                try: 
                    self.model.module.simple3dpose.eval()
                    self.model.module.adaptive_A.train()
                    self.model.module.model_diff.train()                
                except:
                    self.model.simple3dpose.eval()
                    self.model.adaptive_A.train()
                    self.model.model_diff.train()    
        else:
            self.model.train()

        metric_dict = defaultdict(list)

        lr_check(self.optimizer, epoch, master)

        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator, dynamic_ncols=True) if master else self.batch_generator
        for i, meta in enumerate(batch_generator):
            for k, _ in meta.items():
                meta[k] = meta[k].cuda()
                if k == 'img':
                    meta[k] = meta[k].requires_grad_()

            imgs = meta['img'].cuda()
            inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
            gt_pose_root = meta['root_cam'].cuda()
            depth_factor = meta['depth_factor'].cuda()
            flip_mask = meta['flip'].cuda().reshape(-1) if cfg.aug.flip else None
            batch_size = gt_pose_root.shape[0]

            joint_uvd_valid, joint_cam_valid, mesh_valid = meta['joint_uvd_valid'].cuda(), meta['joint_cam_valid'].cuda(), meta['mesh_valid'].cuda()
            gt_pose, gt_uvd_pose, gt_mesh = meta['joint_cam'].cuda(), meta['joint_uvd'].cuda(), meta['mesh_cam'].cuda()

            gt_uvd_base = gt_uvd_pose.clone().reshape(batch_size, -1, 3)    # (B, J, 3)
            
            _, pred_uvd_pose, _, confidence, pred_mesh, _, pred_heatmap2d, diff_pred, diff_gt, diff_pred_uvd_pose, pred_uv_pose = self.model(imgs, inv_trans, intrinsic_param, gt_pose_root, depth_factor, gt_uvd_base.clone(), flip_item=None, flip_mask=flip_mask, is_train=True)

            # ############## diff_model loss ##############
            noise_loss = self.loss['noise'](diff_pred, diff_gt, joint_uvd_valid)

            metric_dict['noise_loss'].append(noise_loss.item())
            
            loss = cfg.loss.loss_weight_noise * noise_loss

            # ############## mesh loss ##############
            pred_pose = torch.cat((torch.matmul(self.J_regressor, pred_mesh), torch.matmul(self.vm_B.T[None], pred_mesh[:, self.selected_indices])), dim=1)

            joint3d_loss = self.loss['joint_3d'](pred_uvd_pose, gt_uvd_pose, joint_uvd_valid)
            joint3d_reg_loss = self.loss['joint_reg_3d'](pred_pose, gt_pose, joint_cam_valid)
            conf_loss = self.loss['conf'](confidence, joint_uvd_valid)
            mesh3d_loss = self.loss['mesh_3d'](pred_mesh, gt_mesh, mesh_valid)
            mesh3d_normal_loss = self.loss['normal'](pred_mesh, gt_mesh)
            mesh3d_edge_loss = self.loss['edge'](pred_mesh, gt_mesh)

            metric_dict['joint3d_loss'].append(joint3d_loss.item())
            metric_dict['joint3d_reg_loss'].append(joint3d_reg_loss.item())
            metric_dict['conf_loss'].append(conf_loss.item())
            metric_dict['mesh3d_loss'].append(mesh3d_loss.item())
            metric_dict['mesh3d_normal_loss'].append(mesh3d_normal_loss.item())
            metric_dict['mesh3d_edge_loss'].append(mesh3d_edge_loss.item())

            loss = loss + \
                cfg.loss.loss_weight_joint3d * joint3d_loss + \
                cfg.loss.loss_weight_joint3d_reg * joint3d_reg_loss + \
                cfg.loss.loss_weight_conf * conf_loss + \
                cfg.loss.loss_weight_mesh3d * mesh3d_loss + \
                cfg.loss.loss_weight_normal * mesh3d_normal_loss
            
            if epoch > self.edge_add_epoch:
                loss = loss + cfg.loss.loss_weight_edge * mesh3d_edge_loss

            # ############## heatmap2d loss ##############
            heatmap2d_loss = self.loss['noise'](diff_gt, diff_gt)     # dummy loss
            if cfg.model.hrnet.heatmap2d and cfg.loss.loss_weight_heatmap2d:
                gt_heatmap2d, gt_heatmap2d_valid = meta['joint_uvd_hm'].cuda(), meta['joint_uvd_hm_valid'].cuda()
                heatmap2d_loss = self.loss['heatmap_2d'](pred_heatmap2d, gt_heatmap2d, gt_heatmap2d_valid)
                metric_dict['heatmap2d_loss'].append(heatmap2d_loss.item())
                loss = loss + (cfg.loss.loss_weight_heatmap2d * heatmap2d_loss)

            metric_dict['total_loss'].append(loss.item())

            if not cfg.model.simple3dpose.fix_network:
                self.optimizer['3d'].zero_grad()
            if not cfg.model.simple3dmesh.fix_network:
                self.optimizer['mesh'].zero_grad()
            self.optimizer['diff'].zero_grad()
            loss.backward()

            if not cfg.model.simple3dpose.fix_network:
                try:
                    # clip the grad
                    torch.nn.utils.clip_grad_norm_(self.model.module.simple3dpose.parameters(), cfg.loss.grad_clip)
                except Exception:
                    torch.nn.utils.clip_grad_norm_(self.model.simple3dpose.parameters(), cfg.loss.grad_clip)
                    pass
                self.optimizer['3d'].step()
            if not cfg.model.simple3dmesh.fix_network:
                try:
                    # clip the grad
                    torch.nn.utils.clip_grad_norm_(self.model.module.adaptive_A.parameters(), cfg.loss.grad_clip)
                except Exception:
                    torch.nn.utils.clip_grad_norm_(self.model.adaptive_A.parameters(), cfg.loss.grad_clip)
                    pass
                self.optimizer['mesh'].step()
            try:
                # clip the grad
                torch.nn.utils.clip_grad_norm_(self.model.module.model_diff.parameters(), cfg.loss.grad_clip)
            except Exception:
                torch.nn.utils.clip_grad_norm_(self.model.model_diff.parameters(), cfg.loss.grad_clip)
                pass
            self.optimizer['diff'].step()

            running_loss += float(loss.detach().item())

            if cfg.diffusion.ema:
                self.ema_helper.update(self.model.module.model_diff)

            if master:
                if i % self.print_freq == 0:
                    batch_generator.set_description(f'Epoch{epoch}_({i}/{len(self.batch_generator)}) => '
                                                    f'3d: {joint3d_loss.detach():.4f}, '
                                                    f'3d_r: {joint3d_reg_loss.detach():.4f}, '
                                                    f'conf: {conf_loss.detach():.4f}, '
                                                    f'mesh: {mesh3d_loss.detach():.4f}, '
                                                    f'norm: {mesh3d_normal_loss.detach():.4f}, '
                                                    f'noise: {noise_loss.detach():.4f}, '
                                                    f'hm2d: {heatmap2d_loss.detach():.4f}, '
                                                    f'total: {loss.detach():.4f} ')

                if i % self.vis_freq == 0:
                    nrow = min(max(batch_size//3, 1), 4)
                    ncol = (min(batch_size//nrow, 3))*2
                    vis_joints_3d(imgs.detach().cpu().numpy(), pred_pose.detach().cpu().numpy(), None, file_name='train_{:08}_vm3d_reg_pred.jpg'.format(i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                    vis_joints_3d(imgs.detach().cpu().numpy(), pred_mesh.detach().cpu().numpy(), None, file_name='train_{:08}_mesh_pred.jpg'.format(i), nrow=nrow, ncol=ncol)
                    vis_joints_3d(imgs.detach().cpu().numpy(), diff_pred_uvd_pose.detach().cpu().numpy(), None, file_name='train_{:08}_uvd_diff_pred.jpg'.format(i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                    vis_joints_3d(imgs.detach().cpu().numpy(), gt_pose.detach().cpu().numpy(), None, file_name='train_{:08}_vm3d_gt.jpg'.format(i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                    vis_joints_3d(imgs.detach().cpu().numpy(), gt_uvd_base.detach().cpu().numpy(), None, file_name='train_{:08}_uvd_gt.jpg'.format(i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                    vis_joints_3d(imgs.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy(), None, file_name='train_{:08}_mesh_gt.jpg'.format(i), nrow=nrow, ncol=ncol)
                    if cfg.model.hrnet.heatmap2d:
                        vis_heatmap_2d(imgs.detach().cpu().numpy(), gt_heatmap2d.detach().cpu().numpy(), None, file_name='train_{:08}_hm2d_gt.jpg'.format(i), nrow=nrow)
                        vis_heatmap_2d(imgs.detach().cpu().numpy(), pred_heatmap2d.detach().cpu().numpy(), None, file_name='train_{:08}_hm2d_pred.jpg'.format(i), nrow=nrow)

                for title, value in metric_dict.items():
                    self.writer.add_scalar("{}/{}".format('train', title), value[-1], n_iters_total)
                n_iters_total += 1

        self.loss_history.append(running_loss / len(self.batch_generator))

        logging.info(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')
        return n_iters_total

class Diffusion3DMeshTester:
    def __init__(self, args, load_path='', writer=None, master=None):
        self.val_loader_list, self.val_dataset, self.model, _, _, _, _, _, _, _, _ = \
            prepare_network(args, load_path=load_path, is_train=False, master=master)

        self.joint_num = self.val_dataset[0].joint_num
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.print_freq = cfg.train.print_freq
        self.vis_freq = cfg.test.vis_freq
        self.writer = writer
        self.device = args.device

        self.J_regressor = torch.Tensor(self.val_dataset[0].joint_regressor).cuda()
        self.selected_indices = self.val_dataset[0].selected_indices
        self.vm_B = torch.Tensor(self.val_dataset[0].vm_B).cuda()

        self.flip_test = cfg.test.flip_test

    def test(self, epoch, master, world_size, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        eval_prefix = f'Epoch{epoch} ' if epoch else 'Test '
        for dataset_name, val_dataset, val_loader in zip(cfg.dataset.test_list, self.val_dataset, self.val_loader_list):
            results = defaultdict(list)
            metric_dict = defaultdict(list)

            joint_error = 0.0
            logging.info(f'=> Evaluating on {dataset_name} ...')
            loader = tqdm(val_loader, dynamic_ncols=True) if master else val_loader
            with torch.no_grad():
                for i, meta in enumerate(loader):
                    imgs = meta['img'].cuda()
                    inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
                    depth_factor, gt_pose_root = meta['depth_factor'].cuda(), meta['root_cam'].cuda()
                    batch_size = gt_pose_root.shape[0]

                    gt_pose, gt_uvd_pose, gt_mesh = meta['joint_cam'].cuda(), meta['joint_uvd'].cuda(), meta['mesh_cam'].cuda()
                    gt_uvd_base = gt_uvd_pose.clone().reshape(batch_size, -1, 3)    # (B, J, 3)

                    if self.flip_test:
                        if isinstance(imgs, list):
                            imgs_flip = [flip_img(img.clone()) for img in imgs]
                        else:
                            imgs_flip = flip_img(imgs.clone())
                    
                    out = defaultdict(list)
                    for midx in range(cfg.test.multi_n):
                        if midx:
                            self.model.module.ddim_sampler.sample_method = 'rand'
                        else:
                            self.model.module.ddim_sampler.sample_method = 'z0'
                        pred_vm, pred_uvd_pose, _, confidence, pred_mesh, pred_inter_mesh_list, _, _, diff_pred_uvd_pose = self.model(imgs.clone(), inv_trans.clone(),intrinsic_param.clone(), gt_pose_root.clone(), depth_factor.clone(), gt_uvd_base.clone(), flip_item=None, flip_mask=None, is_train=False)
                        pred_pose = torch.cat((torch.matmul(self.J_regressor, pred_mesh), torch.matmul(self.vm_B.T[None], pred_mesh[:, self.selected_indices])), dim=1)
                        out['diff_pred_uvd_pose'].append(diff_pred_uvd_pose.detach().cpu().numpy())
                        out['pred_mesh'].append(pred_mesh.detach().cpu().numpy())
                        out['pred_pose'].append(pred_pose.detach().cpu().numpy())
                        if self.flip_test:
                            pred_vm_flip, _, _, _, pred_mesh_flip, pred_inter_mesh_flip_list, _, _, diff_pred_uvd_pose_flip = self.model(imgs_flip.clone(), inv_trans.clone(), intrinsic_param.clone(), gt_pose_root.clone(), depth_factor.clone(), gt_uvd_base.clone(), flip_item=(pred_uvd_pose.clone(), confidence.clone()), flip_output=True, flip_mask=None, is_train=False)
                            pred_pose_flip = torch.cat((torch.matmul(self.J_regressor, pred_mesh_flip), torch.matmul(self.vm_B.T[None], pred_mesh_flip[:, self.selected_indices])), dim=1)
                            out['diff_pred_uvd_pose_flip'].append(diff_pred_uvd_pose_flip.detach().cpu().numpy())
                            out['pred_mesh_flip'].append(pred_mesh_flip.detach().cpu().numpy())
                            out['pred_pose_flip'].append(pred_pose_flip.detach().cpu().numpy())
                    for key in out.keys():
                        out[key] = np.stack(out[key],axis=0)
                        results[key].append(out[key])

                    results['gt_pose'].append(gt_pose.detach().cpu().numpy())
                    results['gt_mesh'].append(gt_mesh.detach().cpu().numpy())
                    results['gt_uvd_base'].append(gt_uvd_base.detach().cpu().numpy())
                    results['gt_pose_root'].append(gt_pose_root.detach().cpu().numpy())
                    results['focal_l'].append(meta['focal_l'].detach().cpu().numpy())
                    results['center_pt'].append(meta['center_pt'].detach().cpu().numpy())
                    results['idx'].append(meta['idx'].detach().cpu().numpy())
                    
                    if master:
                        if i % self.print_freq == 0:
                            loader.set_description(f'{eval_prefix}({i}/{len(val_loader)}) => ')
                        if cfg.test.vis and i % self.vis_freq == 0:
                            nrow = min(max(batch_size//3, 1), 4)
                            ncol = (min(batch_size//nrow, 3))*2
                            for midx in range(0, cfg.test.multi_n, 4):
                                if midx > 10:
                                    break
                                if self.flip_test:
                                    vis_joints_3d(imgs.detach().cpu().numpy(), out['pred_pose_flip'][midx], None, file_name='val_{}_{:08}_vm3d_reg_pred_flip_hypo{:03d}.jpg'.format(dataset_name, i, midx), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                                    vis_joints_3d(imgs.detach().cpu().numpy(), out['pred_mesh_flip'][midx], None, file_name='val_{}_{:08}_mesh_pred_flip_hypo{:03d}.jpg'.format(dataset_name, i, midx), nrow=nrow, ncol=ncol)
                                    vis_joints_3d(imgs.detach().cpu().numpy(), out['pred_vm_flip'][midx], None, file_name='val_{}_{:08}_vm3d_pred_flip_hypo{:03d}.jpg'.format(dataset_name, i, midx), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                                    vis_joints_3d(imgs.detach().cpu().numpy(), out['diff_pred_uvd_pose_flip'][midx], None, file_name='val_{}_{:08}_uvd_diff_pred_flip_hypo{:03d}.jpg'.format(dataset_name, i, midx), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                                vis_joints_3d(imgs.detach().cpu().numpy(), out['pred_pose'][midx], None, file_name='val_{}_{:08}_vm3d_reg_pred_hypo{:03d}.jpg'.format(dataset_name, i, midx), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                                vis_joints_3d(imgs.detach().cpu().numpy(), out['pred_mesh'][midx], None, file_name='val_{}_{:08}_mesh_pred_hypo{:03d}.jpg'.format(dataset_name, i, midx), nrow=nrow, ncol=ncol)
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_pose.detach().cpu().numpy(), None, file_name='val_{}_{:08}_vm3d_gt.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=nrow, ncol=ncol)
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_gt.jpg'.format(dataset_name, i), nrow=nrow, ncol=ncol)
                dist.barrier()

                end = time.time()
                keys_list = ['pred_mesh','pred_pose']
                if self.flip_test:
                    keys_list += ['pred_mesh_flip','pred_pose_flip']
                for term in results.keys():
                    if term in keys_list:
                        results[term] = np.concatenate(results[term],axis=1)
                    else:
                        results[term] = np.concatenate(results[term])
                print(f'==> concatenating results done, using {(time.time()-end):.2f}s')

                if master:
                    end = time.time()
                    msg = f'\n{eval_prefix}'
                    if self.flip_test:
                        msg_flip = f'\n{eval_prefix} Flip_test'
                    if cfg.test.method == 'min':
                        for midx in range(cfg.test.multi_n):
                            joint_error_dict = val_dataset.compute_per_joint_err_dict(results['pred_pose'][midx], results['gt_pose'],mean=False)
                            mesh_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh'][midx], results['gt_mesh'],\
                                results['pred_pose'][midx], results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name, mean=False, midx=midx)
                            for metric_key in mesh_error_dict.keys():
                                metric_dict[metric_key].append(mesh_error_dict[metric_key])
                            for metric_key in joint_error_dict.keys():
                                metric_dict[metric_key].append(joint_error_dict[metric_key])
                            if self.flip_test:
                                joint_flip_error_dict = val_dataset.compute_per_joint_err_dict(results['pred_pose_flip'][midx], results['gt_pose'],mean=False)
                                mesh_flip_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh_flip'][midx], results['gt_mesh'],\
                                    results['pred_pose_flip'][midx], results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name, mean=False, midx=midx, is_flip=True)
                                for metric_key in mesh_error_dict.keys():
                                    metric_dict[metric_key+'_flip'].append(mesh_flip_error_dict[metric_key])
                                for metric_key in joint_error_dict.keys():
                                    metric_dict[metric_key+'_flip'].append(joint_flip_error_dict[metric_key])
                    elif cfg.test.method == 'mean':
                        joint_error_dict = val_dataset.compute_per_joint_err_dict(results['pred_pose'].mean(axis=0), results['gt_pose'])
                        mesh_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh'].mean(axis=0), results['gt_mesh'],\
                            results['pred_pose'].mean(axis=0), results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name)
                        for metric_key in mesh_error_dict.keys():
                            metric_dict[metric_key].append(mesh_error_dict[metric_key].tolist())
                        for metric_key in joint_error_dict.keys():
                            metric_dict[metric_key].append(joint_error_dict[metric_key].tolist())
                        if self.flip_test:
                            joint_flip_error_dict = val_dataset.compute_per_joint_err_dict(results['pred_pose_flip'].mean(axis=0), results['gt_pose'])
                            mesh_flip_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh_flip'].mean(axis=0), results['gt_mesh'],\
                                results['pred_pose_flip'].mean(axis=0), results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name, is_flip=True)
                            for metric_key in mesh_error_dict.keys():
                                metric_dict[metric_key+'_flip'].append(mesh_flip_error_dict[metric_key].tolist())
                            for metric_key in joint_error_dict.keys():
                                metric_dict[metric_key+'_flip'].append(joint_flip_error_dict[metric_key].tolist())
                    print(f'==> [only master] computing metrics for all hypos done, using {time.time()-end:.2f}s')
                    end = time.time()
                    for metric_key in metric_dict.keys():
                        if cfg.test.method == 'min':
                            metric_dict[metric_key] = np.stack(metric_dict[metric_key],axis=0)
                            metric_dict[metric_key] = np.min(metric_dict[metric_key],axis=0).mean().tolist()
                        elif cfg.test.method == 'mean':
                            metric_dict[metric_key] = np.array(metric_dict[metric_key]).mean()
                        if 'flip' in metric_key:
                            msg_flip += f' | {metric_key:12}: {metric_dict[metric_key]:3.2f}'
                        else:
                            msg += f' | {metric_key:12}: {metric_dict[metric_key]:3.2f}'
                    if self.flip_test:
                        msg += f'{msg_flip}'
                    msg += '\n'
                    print(f'==> [only master] aggregating metrics done, using {time.time()-end:.2f}s')
                    print(msg)
                        
                    if cfg.test.save_result:
                        result_path = osp.join(cfg.metric_dir, "{}_result_valset.pkl".format(dataset_name))
                        with open(result_path, 'wb') as f:
                            save_data = {
                                'idx': results['idx'],
                                'f':results['focal_l'],
                                'c':results['center_pt'],
                                'gt_mesh': results['gt_mesh'],
                                'gt_pose': results['gt_pose'],
                                'pred_vm': results['pred_vm'],
                                'pred_mesh': results['pred_mesh'],
                                'pred_pose': results['pred_pose'],
                            }
                            if self.flip_test:
                                save_data.update({
                                    'pred_vm_flip': results['pred_vm_flip'],
                                    'pred_mesh_flip': results['pred_mesh_flip'],
                                    'pred_pose_flip': results['pred_pose_flip'],
                                })
                            pickle.dump(save_data, f)

                    for title, value in metric_dict.items():
                        self.writer.add_scalar("{}_{}/{}_epoch".format('val', dataset_name, title), value, epoch)
                    
                    # saving metric
                    metric_path = osp.join(cfg.metric_dir, "{}_metric_e{}_valset.json".format(dataset_name, epoch))
                    with open(metric_path, 'w') as fout:
                        json.dump(metric_dict, fout, indent=4, sort_keys=True)
                    print(f'=> writing metric dict to {metric_path}')
