import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
import subprocess
import glob
from collections import defaultdict
import imageio
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from termcolor import colored
from torchvision import transforms
from torch.utils.data import DataLoader

from vmpro.core.config import cfg, update_config, init_experiment_dir
from vmpro.dataset.demo_dataset import DemoDataset
from vmpro.utils.vis import render_mesh
from vmpro.utils.funcs_utils import load_checkpoint
from vmpro.utils.smpl_utils import get_smpl_faces
from vmpro.utils.coord_utils import pixel2cam
from vmpro.utils.diff_utils import get_neighbour_matrix
from vmpro.utils.ema_utils import EMAHelper
import vmpro.models as models

# detection module
from virtualpose.core.config import config as det_cfg
from virtualpose.core.config import update_config as det_update_config
from virtualpose.utils.transforms import inverse_affine_transform_pts_cuda
from virtualpose.utils.utils import load_backbone_validate
import virtualpose.models as det_models
import virtualpose.dataset as det_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Infer VMarker-Pro')

    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--cfg', type=str, help='experiment configure file name')
    parser.add_argument('--device', type=str, default='0', help='assign multi-devices by comma concat')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--experiment_name', type=str, default='', help='experiment name')
    parser.add_argument('--data_path', type=str, default='.', help='data dir path')
    parser.add_argument('--cur_path', type=str, default='.', help='current dir path')

    parser.add_argument('--input_type', default='image', choices=['image', 'video'], help='input type')
    parser.add_argument('--input_path', default='inputs/input.mp4', help='path to the input data')
    parser.add_argument('--save_vid', action='store_true', help='save rendered video')
    parser.add_argument('--test_fps', action='store_true', help='test fps')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for detection and motion capture')
    args = parser.parse_args()
    return args

def write_obj(path,point,face):
    with open(path,'w') as fid:
        for p in point:
            fid.write('v {} {} {}\n'.format(str(p[0]),str(p[1]),str(p[2])))
        for f in face:
            fid.write('f {} {} {}\n'.format(str(f[0]+1),str(f[1]+1),str(f[2]+1)))

def time_sync():
    '''Waits for all kernels in all streams on a CUDA device to complete if cuda is available.'''
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class Diff3DMeshInferencer:
    def __init__(self, args, load_path='', writer=None, img_path_list=[], detection_all=[], max_person=-1, fps=20):
        self.args = args
        # prepare inference dataset
        demo_dataset = DemoDataset(img_path_list, detection_all)
        self.demo_dataset = demo_dataset
        self.detection_all = detection_all
        self.img_path_list = img_path_list
        self.max_person = max_person
        self.fps = fps
        self.demo_dataloader = DataLoader(self.demo_dataset, batch_size=min(args.batch_size, len(detection_all)), num_workers=8)
        if self.args.test_fps:
            self.warmup_dataloader = DataLoader(self.demo_dataset, batch_size=min(args.batch_size, len(detection_all)), num_workers=8)

        # prepare inference model
        vm_A = demo_dataset.vm_A if hasattr(demo_dataset, "vm_A") else None
        selected_indices = demo_dataset.selected_indices if hasattr(demo_dataset, "selected_indices") else None
        if cfg.model.simple3dmesh.noise_reduce:
            self.model = models.simple3dmesh_post.get_model(demo_dataset.vertex_num, demo_dataset.flip_pairs, \
                vm_A=vm_A, \
                selected_indices=selected_indices, \
                mesh_template=demo_dataset.smpl_template)
        elif cfg.model.name == 'diff3dmesh':
            neighbour_matrix = get_neighbour_matrix()
            self.model = models.diff3dmesh.get_model(demo_dataset.vertex_num, demo_dataset.flip_pairs, \
                vm_A=vm_A, \
                selected_indices=selected_indices, \
                neighbour_matrix=neighbour_matrix)
            if cfg.diffusion.ema:
                ema_helper = EMAHelper(mu=cfg.diffusion.ema_rate)
                ema_helper.register(self.model.model_diff.cuda())
        else:
            self.model = models.simple3dmesh.get_model(demo_dataset.vertex_num, demo_dataset.flip_pairs, \
                vm_A=vm_A, \
                selected_indices=selected_indices)

        # load weight 
        if load_path != '':
            print('==> Loading checkpoint')
            checkpoint = load_checkpoint(load_path, master=True)

            if 'model_state_dict' in checkpoint.keys():
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            try:
                if cfg.model.name == 'simple3dmesh' and cfg.model.simple3dmesh.noise_reduce:
                    self.model.simple3dmesh.load_state_dict(state_dict, strict=True)
                else:
                    self.model.load_state_dict(state_dict, strict=True)
                    if cfg.diffusion.ema:
                        ema_helper.load_state_dict(checkpoint['ema_helper'])
                        if cfg.model.name == 'diff3dmesh':
                            ema_helper.ema(self.model.model_diff)
                print(colored(f'Successfully load checkpoint from {load_path}.', 'green'))
            except:
                print(colored(f'Failed to load checkpoint in {load_path}', 'red'))
        if self.model:
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model)

        # initialize others
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.J_regressor = torch.Tensor(self.demo_dataset.joint_regressor).cuda()
        self.vm_B = torch.Tensor(self.demo_dataset.vm_B).cuda()
        self.selected_indices = self.demo_dataset.selected_indices
        self.smpl_faces = get_smpl_faces()
        self.multi_n = cfg.test.multi_n

    def infer(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        total_time = 0

        for midx in range(self.multi_n):
            results = defaultdict(list)
            if midx:
                self.model.module.ddim_sampler.sample_method = 'rand'
            else:
                self.model.module.ddim_sampler.sample_method = 'z0'
            with torch.no_grad():
                if self.args.test_fps:
                    print(f"===> Warmup For FPS Test...")
                    warmup_batches = 5 # warmup 5 batches
                    batch_count = 0
                    while batch_count < warmup_batches:
                        for i, meta in enumerate(tqdm(self.warmup_dataloader, dynamic_ncols=True)):
                            meta = {key: term.cuda() for key, term in meta.items()}
                            imgs = meta['img']
                            inv_trans, intrinsic_param = meta['inv_trans'], meta['intrinsic_param']
                            pose_root, depth_factor = meta['root_cam'], meta['depth_factor']
                            _, __, _, _, pred_mesh, _,__, pred_root_xy_img,__ = self.model(imgs, inv_trans, intrinsic_param, pose_root, depth_factor, flip_item=None, flip_mask=None,is_train=False)
                            batch_count += 1
                            if batch_count >= warmup_batches: break

                for i, meta in enumerate(tqdm(self.demo_dataloader, dynamic_ncols=True)):
                    for k, _ in meta.items():
                        meta[k] = meta[k].cuda()

                    imgs = meta['img'].cuda()
                    inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
                    pose_root = meta['root_cam'].cuda()-500
                    depth_factor = meta['depth_factor'].cuda()
                    end = time_sync()
                    _, __, _, _, pred_mesh, _,__, pred_root_xy_img,__ = self.model(imgs, inv_trans, intrinsic_param, pose_root, depth_factor, flip_item=None, flip_mask=None,is_train=False)
                    total_time += (time_sync() - end)
                    results['pred_mesh'].append(pred_mesh.detach().cpu().numpy())
                    results['pose_root'].append(pose_root.detach().cpu().numpy())
                    results['pred_root_xy_img'].append(pred_root_xy_img.squeeze(1).squeeze(-1).detach().cpu().numpy())
                    results['focal_l'].append(meta['focal_l'].detach().cpu().numpy())
                    results['center_pt'].append(meta['center_pt'].detach().cpu().numpy())

            for term in results.keys():
                results[term] = np.concatenate(results[term])
            results['midx'] = midx
            self.visualize(results)
        if self.args.test_fps:
            fps = len(self.img_path_list)/(total_time*self.multi_n)
            print(f'When denoising step is {cfg.diffusion.ddim_steps}, FPS is {fps}')
        return results

    def visualize(self, results):
        pred_mesh = results['pred_mesh']  # (N*T, V, 3)
        pred_root_xy_img = results['pred_root_xy_img']  # (N*T, J, 2)
        pose_root = results['pose_root']  # (N*T, 3)
        focal_l = results['focal_l']
        center_pt = results['center_pt']
        midx = results['midx']
        # root modification (differenct root definition betwee VM & VirtualPose)
        new_pose_root = []
        for root_xy, root_cam, focal, center in zip(pred_root_xy_img, pose_root, focal_l, center_pt):
            root_img = np.array([root_xy[0], root_xy[1], root_cam[-1]])
            new_root_cam = pixel2cam(root_img[None,:], center, focal)
            new_pose_root.append(new_root_cam)
        pose_root = np.array(new_pose_root)  # (N*T, 1, 3)

        pred_mesh = pred_mesh + pose_root
        if cfg.test.save_obj:
            for idx in range(pred_mesh.shape[0]):
                write_obj(osp.join(cfg.vis_dir, f"{self.args.input_path.split('/')[-1][:-4]}_hypo_{midx}_item_{idx}.obj"),pred_mesh[idx],self.smpl_faces)
        pred_mesh_T_N = np.zeros((len(self.img_path_list), self.max_person, self.demo_dataset.vertex_num, 3))
        videowriter = imageio.get_writer(osp.join(cfg.vis_dir, f"{self.args.input_path.split('/')[-1][:-4]}_results_in_2d_{midx}.mp4"), fps=self.fps)
        for img_idx, img_path in enumerate(tqdm(self.img_path_list, dynamic_ncols=True)):
            img = cv2.imread(img_path)
            ori_img_height, ori_img_width = img.shape[:2]

            chosen_mask = self.detection_all[:, 0] == img_idx
            pred_mesh_T = pred_mesh[chosen_mask]  # (N, V, 3)
            focal_T = focal_l[chosen_mask]  # (N, ...)
            center_pt_T = center_pt[chosen_mask]  # (N, ...)
            pred_mesh_T_N[img_idx, :pred_mesh_T.shape[0]] = pred_mesh_T

            # render to image
            try:
                rgb, depth = render_mesh(ori_img_height, ori_img_width, pred_mesh_T/1000.0, self.smpl_faces, {'focal': focal_T, 'princpt': center_pt_T})
                valid_mask = (depth > 0)[:,:,None] 
                rendered_img = rgb * valid_mask + img[:,:,::-1] * (1-valid_mask)
                cv2.imwrite(osp.join(cfg.vis_dir, f"{self.args.input_path.split('/')[-1][:-4]}_results_in_2d_{midx}.jpg"), rendered_img.astype(np.uint8)[...,::-1])
                if self.args.save_vid:
                    videowriter.append_data(rendered_img.astype(np.uint8))
            except:
                if self.args.save_vid:
                    videowriter.append_data(img.astype(np.uint8)[...,::-1])
        if self.args.save_vid:
            videowriter.close()


def output2original_scale(meta, output, vis=False):
    img_paths, trans_batch = meta['image'], meta['trans']
    bbox_batch, depth_batch, roots_2d = output['bboxes'], output['depths'], output['roots_2d']

    scale = torch.tensor((det_cfg.NETWORK.IMAGE_SIZE[0] / det_cfg.NETWORK.HEATMAP_SIZE[0], \
                        det_cfg.NETWORK.IMAGE_SIZE[1] / det_cfg.NETWORK.HEATMAP_SIZE[1]), \
                        device=bbox_batch.device, dtype=torch.float32)
    
    det_results = []
    valid_frame_idx = []
    max_person = 0
    for i, img_path in enumerate(img_paths):
        if vis:
            img = cv2.imread(img_path)
        if args.input_type == "image":
            frame_id = 0
        else:
            frame_id = int(img_path.split('/')[-1][:-4])-1
        trans = trans_batch[i].to(bbox_batch[i].device).float()

        n_person = 0
        for bbox, depth, root_2d in zip(bbox_batch[i], depth_batch[i], roots_2d[i]):
            if torch.all(bbox == 0):
                break
            bbox = (bbox.view(-1, 2) * scale[None, [1, 0]]).view(-1)
            root_2d *= scale[[1, 0]]
            bbox_origin = inverse_affine_transform_pts_cuda(bbox.view(-1, 2), trans).reshape(-1)
            roots_2d_origin = inverse_affine_transform_pts_cuda(root_2d.view(-1, 2), trans).reshape(-1)

            # frame_id, x_min, y_min, x_max, y_max, pixel_root_x, pixel_root_y, depth
            det_results.append([frame_id] + bbox_origin.cpu().numpy().tolist() + roots_2d_origin.cpu().numpy().tolist() + depth.cpu().numpy().tolist())

            if vis:
                img = cv2.putText(img, '%.2fmm'%depth, (int(bbox_origin[0]), int(bbox_origin[1] - 5)),\
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                img = cv2.rectangle(img, (int(bbox_origin[0]), int(bbox_origin[1])), (int(bbox_origin[2]), int(bbox_origin[3])), \
                    (255, 0, 0), 1)
                img = cv2.circle(img, (int(roots_2d_origin[0]), int(roots_2d_origin[1])), 5, (0, 0, 255), -1)
            n_person += 1

        if vis:
            cv2.imwrite(f'{cfg.vis_dir}/origin_det_{i}.jpg', img)
        max_person = max(n_person, max_person)
        if n_person:
            valid_frame_idx.append(frame_id)
    return det_results, max_person, valid_frame_idx

def detect_all_persons(args, img_dir):

    # prepare detection model
    virtualpose_name = 'VirtualPose' 
    det_update_config(f'{virtualpose_name}/configs/images/images_inference.yaml')

    det_model = eval('det_models.multi_person_posenet.get_multi_person_pose_net')(det_cfg, is_train=False)
    with torch.no_grad():
        det_model = torch.nn.DataParallel(det_model.cuda())

    pretrained_file = osp.join(args.cur_path, f'{virtualpose_name}', det_cfg.NETWORK.PRETRAINED)
    state_dict = torch.load(pretrained_file)
    new_state_dict = {k:v for k, v in state_dict.items() if 'backbone.pose_branch.' not in k}
    det_model.module.load_state_dict(new_state_dict, strict = False)
    pretrained_file = osp.join(args.cur_path, f'{virtualpose_name}', det_cfg.NETWORK.PRETRAINED_BACKBONE)
    det_model = load_backbone_validate(det_model, pretrained_file)

    # prepare detection dataset
    infer_dataset = det_dataset.images(
        det_cfg, img_dir, focal_length=1700, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
        ]))

    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=det_cfg.TEST.BATCH_SIZE * int(args.gpus),
        shuffle=False,
        num_workers=4,
        pin_memory=True)
    
    det_model.eval()

    max_person = 0
    detection_all = []
    valid_frame_idx_all = []
    with torch.no_grad():
        for _, (inputs, targets_2d, weights_2d, targets_3d, meta, input_AGR) in enumerate(tqdm(infer_loader, dynamic_ncols=True)):
            _, _, output, _, _ = det_model(views=inputs, meta=meta, targets_2d=targets_2d,
                                                            weights_2d=weights_2d, targets_3d=targets_3d, input_AGR=input_AGR)
            det_results, n_person, valid_frame_idx = output2original_scale(meta, output)
            detection_all += det_results
            valid_frame_idx_all += valid_frame_idx
            max_person = max(n_person, max_person)

    # list to array
    detection_all = np.array(detection_all)  # (N*T, 8)
    return detection_all, max_person, valid_frame_idx_all

def video_to_images(vid_file, img_folder=None):
    cap = cv2.VideoCapture(vid_file)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    try:
        command = ['ffmpeg',
                '-i', vid_file,
                '-r', str(fps),
                '-f', 'image2',
                '-v', 'error',
                f'{img_folder}/%06d.png']
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command)
    except:
        # 未安装ffmpeg
        frame_count = 0
        
        # 逐帧读取视频并保存为图片
        while True:
            # 读取一帧视频
            ret, frame = cap.read()
            
            # 如果无法读取帧，则退出循环
            if not ret:
                break
            
            # 生成输出文件名
            output_file = "%s/%06d.jpg"%(img_folder, frame_count)
            
            # 保存帧为图片
            cv2.imwrite(output_file, frame)
            
            frame_count += 1
        
        # 释放视频对象
        cap.release()

    print(f'Images saved to \"{img_folder}\"')
    
    
    return fps

def get_image_path(args):

    if args.input_type == "image":
        img_dir = osp.join(osp.dirname(osp.abspath(args.input_path)), args.input_path.split('/')[-1][:-4])
        # try:
        #     shutil.rmtree(img_dir)
        # except:
        #     pass
        os.makedirs(img_dir, exist_ok=True)
        shutil.copy(args.input_path, img_dir)
        img_path_list = [osp.join(img_dir, args.input_path.split('/')[-1])]
        fps = 1
    elif args.input_type == "video":
        basename = osp.basename(args.input_path).split('.')[0]
        img_dir = osp.join(osp.dirname(osp.abspath(args.input_path)), basename)
        # try:
        #     shutil.rmtree(img_dir)
        # except:
        #     pass
        os.makedirs(img_dir, exist_ok=True)
        fps = video_to_images(args.input_path, img_folder=img_dir)

        # get all image paths
        img_path_list = glob.glob(osp.join(img_dir, '*.jpg'))
        img_path_list.extend(glob.glob(osp.join(img_dir, '*.png')))
        img_path_list.sort()
    else:
        assert 0, 'only support image/video input type'
    return img_path_list, img_dir, fps

def main(args):
    # ############ prepare environments ############
    args.is_distributed = False
    args.device = [i for i in range(args.gpus)]

    # set up experiment
    print("args: {}".format(args))
    writer = init_experiment_dir(args.cur_path, args.data_path, args.experiment_name)
    # update config
    if args.cfg:
        update_config(args.cfg)
    torch.manual_seed(args.seed)

    # ############ get all image paths ############
    print("Input path:", args.input_path)
    print("Input type:", args.input_type)
    img_path_list, img_dir, fps = get_image_path(args)

    # ############ detect all persons with estimated root depth ############
    detection_all, max_person, valid_frame_idx_all = detect_all_persons(args, img_dir)

    # ############ prepare virtual marker model ############
    # create the model instance and load checkpoint
    load_path_test = cfg.test.weight_path
    inferencer = Diff3DMeshInferencer(args, load_path=load_path_test, writer=writer, img_path_list=img_path_list, detection_all=detection_all, max_person=max_person, fps=fps)


    # ############ inference virtual marker model ############
    print(f"===> Start inferencing...")
    inferencer.infer(epoch=0)
    print(colored(f'Results saved to {cfg.vis_dir}.', 'green'))
    print(f"===> Done.")


if __name__ == '__main__':
    args = parse_args()
    main(args)
