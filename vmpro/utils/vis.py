import os
import os.path as osp
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from termcolor import colored
try:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    import pyrender
except:
    print(colored('pyrender is not correctly imported.', 'red'))
from mpl_toolkits.mplot3d import Axes3D
import math
import cv2
import trimesh
import torchvision
from vmpro.core.config import cfg

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
CONNECTIVITY_DICT = {
    "human36m": [(0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)],
}
my_green = (135, 153, 124)
my_lightgreen = (187, 225, 163)
my_purple = (185, 190, 235)
my_orange = (230, 147, 115)
my_darkblue = (34, 78, 102)

COLOR_DICT = {
    'human36m': [
        my_lightgreen, my_lightgreen,  # body
        my_lightgreen, my_lightgreen,  # head
        my_purple, my_purple, my_purple,  # left leg        
        my_orange, my_orange, my_orange,  # right leg (green)
        my_orange, my_orange, my_orange,   # right arm
        my_purple, my_purple, my_purple   # left arm
    ],
}


def denormalize_image(image):
    """Reverse to normalize_image() function"""
    return np.clip(255*(image * IMAGENET_STD + IMAGENET_MEAN), 0, 255)

def vis_joints_2d(batch_image, batch_joints, batch_joints_vis, file_name, draw_skeleton=False, batch_image_path=None, batch_trans=None, ncol=10, size=5, padding=2, dataset_name='human36m'):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 2],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    assert batch_joints.shape[2] == 2, 'check batch_joints'
    plt.close('all')
    fig = plt.figure(figsize=(ncol*size, 1*size))
    connectivity = CONNECTIVITY_DICT[dataset_name]

    for col in range(ncol):
        if isinstance(batch_image, np.ndarray):
            img = batch_image[col]
            img = denormalize_image(np.transpose(img.copy(), (1,2,0))).astype(np.uint8)   # C*H*W -> H*W*C
        else:
            img = cv2.imread(batch_image_path[col], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).copy().astype(np.uint8)
            img = cv2.warpAffine(img, batch_trans[col], (cfg.model.input_shape[1], cfg.model.input_shape[0]), flags=cv2.INTER_LINEAR)
            img = img[..., ::-1]

        joints = batch_joints[col]
        joints_vis = batch_joints_vis[col] if batch_joints_vis is not None else np.ones((joints.shape[0], 1))
        joints_scatter = joints * joints_vis
        ax = fig.add_subplot(1, ncol, col + 1)
        ax.imshow(img)
        ax.scatter(joints_scatter[:, 0], joints_scatter[:, 1], s=1, c='yellow', edgecolors='yellow')
        if draw_skeleton:
            for i, jt in enumerate(connectivity):
                xs, ys = [np.array([joints[jt[0], j], joints[jt[1], j]]) for j in range(2)]
                ax.plot(xs, ys, lw=1, ls='-', c='blue', solid_capstyle='round')

    save_path = osp.join(cfg.vis_dir, file_name)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.5)
    plt.show()

    plt.close('all')
    return

def vis_joints_3d(batch_image, batch_joints, batch_joints_vis, file_name, draw_skeleton=False, batch_image_path=None, batch_trans=None, nrow=4, ncol=6, size=5, padding=2, dataset_name='human36m'):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 3],
    }
    '''
    batch_size = batch_image.shape[0]
    batch_joints = batch_joints.reshape(batch_size, -1, 3)
    plt.close('all')
    fig = plt.figure(figsize=(ncol*size, nrow*size))
    connectivity = CONNECTIVITY_DICT[dataset_name]

    for row in range(nrow):
        for col in range(ncol):
            batch_idx = col//2 + row*(ncol//2)
            if isinstance(batch_image, np.ndarray):
                img = batch_image[batch_idx]
                img = denormalize_image(np.transpose(img.copy(), (1,2,0))).astype(np.uint8)   # C*H*W -> H*W*C
            else:
                img = cv2.imread(batch_image_path[batch_idx], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).copy().astype(np.uint8)
                img = cv2.warpAffine(img, batch_trans[batch_idx], (cfg.model.input_shape[1], cfg.model.input_shape[0]), flags=cv2.INTER_LINEAR)
                img = img[..., ::-1]

            joints = batch_joints[batch_idx]
            joints_vis = batch_joints_vis[batch_idx] if batch_joints_vis is not None else np.ones((joints.shape[0], 1))
            joints = joints * joints_vis
            if col%2 == 0:  # draw image
                ax = fig.add_subplot(nrow, ncol, row * ncol + col + 1)
                ax.imshow(img)
            else:   # draw 3d joints
                ax = fig.add_subplot(nrow, ncol, row * ncol + col + 1, projection='3d')
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=25, c=np.array([my_darkblue])/255, edgecolors=np.array([my_darkblue])/255)

                # keep ratio
                X, Y, Z = joints[:, 0], joints[:, 1], joints[:, 2]
                max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

                mid_x = (X.max()+X.min()) * 0.5
                mid_y = (Y.max()+Y.min()) * 0.5
                mid_z = (Z.max()+Z.min()) * 0.5
                if max_range:
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                ax.view_init(elev=-90, azim=-90)
                if draw_skeleton:
                    for i, jt in enumerate(connectivity):
                        xs, ys, zs = [np.array([joints[jt[0], j], joints[jt[1], j]]) for j in range(3)]
                        color = COLOR_DICT[dataset_name][i]
                        if color == (0,139,139): # green
                            set_zorder = 2
                        elif color == (255,215,0): # yellow
                            set_zorder = 2
                        else:
                            set_zorder = 1
                        color = np.array(color) / 255

                        ax.plot(xs, ys, zs, lw=5, ls='-', c=color, zorder=set_zorder, solid_capstyle='round')

    background_color = np.array([252, 252, 252]) / 255

    ax.w_xaxis.set_pane_color(background_color)
    ax.w_yaxis.set_pane_color(background_color)
    ax.w_zaxis.set_pane_color(background_color)

    # # Get rid of the ticks
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    save_path = osp.join(cfg.vis_dir, file_name)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.5)
    plt.show()

    plt.close('all')
    return


def vis_heatmap_2d(batch_image, batch_heatmaps, batch_joints, file_name, batch_image_path=None, batch_trans=None, nrow=4):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: [batch_size, num_joints, height, width]
    file_name: saved file name
    '''

    batch_size, num_joints, heatmap_height, heatmap_width = batch_heatmaps.shape[:4]

    ncol = 1 + 12   # draw 17 joints in one heatmap
    grid_image = np.zeros((nrow * heatmap_height, (1+ncol) * heatmap_width, 3), dtype=np.uint8)

    for row in range(nrow):
        if isinstance(batch_image, np.ndarray):
            image = batch_image[row]
            image = denormalize_image(np.transpose(image.copy(), (1,2,0))).astype(np.uint8)   # C*H*W -> H*W*C
        else:
            image = cv2.imread(batch_image_path[row], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).copy().astype(np.uint8)
            image = cv2.warpAffine(image, batch_trans[row], (cfg.model.input_shape[1], cfg.model.input_shape[0]), flags=cv2.INTER_LINEAR)
        image = image[..., ::-1]
        heatmaps =  np.clip(255*batch_heatmaps[row], 0, 255).astype(np.uint8)

        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * row
        height_end = heatmap_height * (row + 1)
        for col in range(ncol):
            if col == 0:
                heatmap = heatmaps[:17, :, :].max(0)
                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                masked_image = colored_heatmap * 0.7 + resized_image * 0.3
                if batch_joints is not None:
                    for j in range(17):
                        cv2.circle(resized_image, (int(batch_joints[row][j][0]), int(batch_joints[row][j][1])), 2, [0, 0, 255], 1)
                        cv2.circle(masked_image, (int(batch_joints[row][j][0]), int(batch_joints[row][j][1])), 2, [0, 0, 255], 1)
            else:
                heatmap = heatmaps[17+col, :, :]
                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                masked_image = colored_heatmap * 0.7 + resized_image * 0.3
                if batch_joints is not None:
                    cv2.circle(resized_image, (int(batch_joints[row][17+col][0]), int(batch_joints[row][17+col][1])), 2, [0, 0, 255], 1)
                    cv2.circle(masked_image, (int(batch_joints[row][17+col][0]), int(batch_joints[row][17+col][1])), 2, [0, 0, 255], 1)
            width_begin = heatmap_width * (col + 1)
            width_end = heatmap_width * (col + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image
        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    save_path = osp.join(cfg.vis_dir, file_name)
    cv2.imwrite(save_path, grid_image)
    return


def render_mesh(height, width, meshes, face, cam_param):
    # renderer
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # camera
    focal, princpt = cam_param['focal'][0], cam_param['princpt'][0]
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # mesh
    for mesh in meshes:
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

        scene.add(mesh, 'mesh')

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    renderer.delete()
    return rgb, depth