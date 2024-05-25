import numpy as np
import cv2
import random
from vmpro.core.config import cfg 
import math
import torch
import torch.nn.functional as F

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:,:,::-1]
    # if not isinstance(img, np.ndarray):
    #     raise IOError("Fail to read %s" % path)

    return img

def im_to_torch(img):
    """Transform ndarray image to torch tensor.

    Parameters
    ----------
    img: numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.

    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = torch.from_numpy(img).float()
    if img.max() > 1:
        img /= 255
    return img

def generate_integral_uvd_target(joints_3d, num_joints, patch_height, patch_width, bbox_3d_shape, heatmap_dim='2d'):

    target_weight = np.ones((num_joints, 3), dtype=np.float32)
    target_weight[:, 0] = joints_3d[:, 0, 1]
    target_weight[:, 1] = joints_3d[:, 0, 1]
    target_weight[:, 2] = joints_3d[:, 0, 1]

    target = np.zeros((num_joints, 3), dtype=np.float32)
    target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
    target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
    target[:, 2] = joints_3d[:, 2, 0] / bbox_3d_shape[2]

    # for heatmap 3d
    if heatmap_dim=='2d':
        target_hm, target_hm_weight= generate_heatmap2d(joints_3d[:, :2, 0], joints_3d[:, :2, 1])
    elif heatmap_dim=='3d':
        assert False, 'Not implemented heatmap3d yet'
    else:
        target_hm, target_hm_weight = None, None


    target_weight[target[:, 0] > 0.5] = 0
    target_weight[target[:, 0] < -0.5] = 0
    target_weight[target[:, 1] > 0.5] = 0
    target_weight[target[:, 1] < -0.5] = 0
    target_weight[target[:, 2] > 0.5] = 0
    target_weight[target[:, 2] < -0.5] = 0

    target = target.reshape((-1))
    # target_weight = target_weight.reshape((-1))
    return target, target_weight, target_hm, target_hm_weight

def generate_heatmap2d(joints_2d, joints_vis):
    '''
    :param joints_2d:  [num_joints, 2]
    :param joints_vis: [num_joints, 2]
    :return: target, target_weight (1: visible, 0: invisible)
    '''
    num_joints = joints_2d.shape[0]
    image_size = np.array(cfg.model.input_shape)
    heatmap_size = np.array(cfg.model.heatmap_shape)
    sigma = cfg.model.hrnet.sigma
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints_2d[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints_2d[joint_id][1] / feat_stride[1] + 0.5)
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            target_weight[joint_id] = 0
            continue

        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight