import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vmpro.core.config import cfg

class ConfCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, confidence, joint_valid):
        # confidence N, J
        # joint_valid N, J*3
        batch_size = confidence.shape[0]
        joint_valid = joint_valid.view(batch_size, -1, 3)[:, :, 2]  # N, J

        loss = (joint_valid * (-torch.log(confidence + 1e-6))).mean()

        return loss

class CoordLoss(nn.Module):
    def __init__(self, has_valid=False, reduction='mean'):
        super(CoordLoss, self).__init__()

        self.has_valid = has_valid
        self.criterion = nn.L1Loss(reduction=reduction).cuda()

    def forward(self, pred, target, target_valid=None):
        if self.has_valid:
            pred, target = pred * target_valid, target * target_valid

        loss = self.criterion(pred, target)

        return loss

class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss.mean()

class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss.mean()

class HeatmapMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(HeatmapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, heatmap, target, target_weight):
        batch_size = heatmap.size(0)
        num_joints = heatmap.size(1)

        # heatmap [N, J, 64, 64]
        # target [N, J, 64, 64]
        # target_weight [N, J, 1]

        heatmaps_pred = heatmap.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss_h = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()     # [N, 4096]
            heatmap_gt = heatmaps_gt[idx].squeeze()     # [N, 4096]

            if self.use_target_weight:
                loss_h += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss_h += self.criterion(heatmap_pred, heatmap_gt)

        return loss_h / num_joints

def get_mesh_loss(faces=None):
    loss = CoordLoss(has_valid=True), NormalVectorLoss(faces), EdgeLengthLoss(faces), CoordLoss(has_valid=True), CoordLoss(has_valid=True)
    return loss

class Diff_Loss(nn.Module):
    def __init__(self,loss_type):
        super(Diff_Loss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.criterion =nn.L1Loss().cuda()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss().cuda()
        elif loss_type == 'huber':
            self.criterion = nn.SmoothL1Loss().cuda()
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
    def forward(self, pred, target, target_valid=None):
        if target_valid!=None:
            pred, target = pred * target_valid, target * target_valid

        return self.criterion(pred, target)


def get_diff_loss():
    
    loss_type = cfg.diffusion.loss_type
    if loss_type == 'l1':
        return nn.L1Loss().cuda()
    elif loss_type == 'l2':
        return nn.MSELoss().cuda()
    elif loss_type == 'huber':
        return nn.SmoothL1Loss().cuda()
    else:
        raise NotImplementedError("unknown loss type '{loss_type}'")


def get_loss(faces=None):
    # define loss function (criterion) and optimizer
    criterion_3d = CoordLoss(has_valid=True).cuda()
    criterion_2d = CoordLoss(has_valid=True).cuda()
    criterion_3d_reg = CoordLoss(has_valid=True).cuda()
    criterion_conf = ConfCELoss()
    criterion_mesh = get_mesh_loss(faces)
    criterion_noise = Diff_Loss(cfg.diffusion.loss_type)
    criterion_noise_recon = CoordLoss().cuda()
    criterion_dict = {
        'joint_3d': criterion_3d,
        'joint_reg_3d': criterion_3d_reg,
        'conf': criterion_conf,
        'mesh_3d': criterion_mesh[0],
        'normal': criterion_mesh[1],
        'edge': criterion_mesh[2],
        'noise': criterion_noise,
        'heatmap_2d': HeatmapMSELoss().cuda()
    }
    return criterion_dict
