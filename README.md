<div align="center">

  <h1 align="center">VMarker-Pro: Probabilistic 3D Human Mesh Estimation from Virtual Markers</h1>
  
</div>

<div align="left">

  <a>![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)</a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/ShirleyMaxx/VMarker-Pro/blob/main/LICENSE">![License](https://img.shields.io/github/license/metaopt/torchopt?label=license&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=)</a>

</div>


<p align="center">
  <img src="demo/quality_results.png"/>
</p>
<p align="middle">
  <img src="demo/demo1.gif" height="100" /> 
  <img src="demo/demo2.gif" height="100" /> 
</p>
<p align="middle">
  <img src="demo/demo3.gif" height="100" /> 
  <img src="demo/demo4.gif" height="100" /> 
</p>


## Introduction

This is the offical [Pytorch](https://pytorch.org/) implementation of our paper:
<h3 align="center">VMarker-Pro: Probabilistic 3D Human Mesh Estimation from Virtual Markers</h3>

<h4 align="center" style="text-decoration: none;">
  <a href="https://shirleymaxx.github.io/", target="_blank"><b>Xiaoxuan Ma</b></a>
  ,
  <a href="https://scholar.google.com/citations?user=DoUvUz4AAAAJ&hl=en", target="_blank"><b>Jiajun Su</b></a>
  ,
  <a href="https://xy02-05.github.io/", target="_blank"><b>Yuan Xu</b></a>
  ,
  <a href="https://wentao.live/", target="_blank"><b>Wentao Zhu</b></a>
  ,
  <a href="https://www.chunyuwang.org/", target="_blank"><b>Chunyu Wang</b></a>
  ,
  <a href="https://cfcs.pku.edu.cn/english/people/faculty/yizhouwang/index.htm", target="_blank"><b>Yizhou Wang</b></a>

</h4>

It is an extension of [VMarker (CVPR 2023)](https://github.com/ShirleyMaxx/VirtualMarker) which supports probabilistic 3D human mesh estimation.
Below is the overall VMarker-Pro framework.

<p align="center">
  <img src="demo/pipeline.png" width="70%" /> 
</p>

### Notes :dart:
This repository provides guidelines for VMarker-Pro exclusively. It is backward compatible and supports all [VMarker](https://github.com/ShirleyMaxx/VirtualMarker) commands. For instructions on training and inference with VMarker, please refer to [VMarker](https://github.com/ShirleyMaxx/VirtualMarker). The only difference is that this repo supports PyTorch's [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) training.



## Installation

1. Install dependencies. This project is developed using >= python 3.8 on Ubuntu 16.04. NVIDIA GPUs are needed. We recommend you use an [Anaconda](https://www.anaconda.com/) virtual environment.

  ```bash
    # 1. Create a conda virtual environment.
    conda create -n pytorch python=3.8 -y
    conda activate pytorch

    # 2. Install PyTorch >= v1.13.0 following [official instruction](https://pytorch.org/). Please adapt the CUDA version to yours.
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117


    # 3. Pull our code.
    git clone https://github.com/ShirleyMaxx/VMarker-Pro.git
    cd VMarker-Pro

    # 4. Install other packages. This project doesn't have any special or difficult-to-install dependencies.
    sh requirements.sh

    #5. Install VMarker-Pro
    python setup.py develop
  ```
2. Prepare SMPL layer. We use [smplx](https://github.com/vchoutas/smplx#installation).

   1. Install `smplx` package by `pip install smplx`. Already done in the first step.
   2. Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl`, and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) (female & male) and [here](http://smplify.is.tue.mpg.de/) (neutral) to `${Project}/data/smpl`. Please rename them as `SMPL_FEMALE.pkl`, `SMPL_MALE.pkl`, and `SMPL_NEUTRAL.pkl`, respectively.
   3. Download others SMPL-related from [Google drive](https://drive.google.com/drive/folders/1LRMo_7raQuSRuUKAvXKSlzlvQJ5C0IHR?usp=share_link) or [Onedrive](https://chinapku-my.sharepoint.com/:f:/g/personal/2101111546_pku_edu_cn/EitToj4t0BlMmKAo6CZT2H8BMmkyAKQBjY6kO5h0htKveA?e=b57zU5) and put them to `${Project}/data/smpl`.
3. Download data following the **Data** section. In summary, your directory tree should be like this

  ```
    ${Project}
    ├── assets
    ├── command
    ├── configs
    ├── data 
    ├── demo 
    ├── experiment 
    ├── inputs 
    ├── vmpro 
    ├── main 
    ├── models 
    ├── README.md
    ├── setup.py
    `── requirements.sh
  ```

  - `assets` contains the body virtual markers in `npz` format. Feel free to use them.
  - `command` contains the running scripts.
  - `configs` contains the configurations in `yml` format.
  - `data` contains soft links to images and annotations directories.
  - `vmpro` contains kernel codes for our method.
  - `main` contains high-level codes for training or testing the network.
  - `models` contains pre-trained weights. Download from [Google drive](https://drive.google.com/drive/folders/1IQ-rvjj_t6ZInlzDfKBNeJ0noOs3fIaW?usp=share_link) or [Onedrive](https://chinapku-my.sharepoint.com/:f:/g/personal/2101111546_pku_edu_cn/EggzlMbPMwtDr8aBb8HNySQBuhh3CtWceRbQMcMdiztDcg?e=YgTfry).
  - *`experiment` will be automatically made after running the code, it contains the outputs, including trained model weights, test metrics, and visualized outputs.
  
## Quick demo :star:

1. **Installation.** Make sure you have finished the above installation successfully. VMarker-Pro does not detect person and only estimates relative pose and mesh, therefore please also install [VirtualPose](https://github.com/wkom/VirtualPose) following its instructions. VirtualPose will detect all the person and estimate their root depths. Download its model weight from [Google drive](https://drive.google.com/drive/folders/1Y8unp_CQnXsWr1WljAgW0rYCMGkru6Ce?usp=share_link) or [Onedrive](https://chinapku-my.sharepoint.com/:f:/g/personal/2101111546_pku_edu_cn/ElPhUt3LaJpMgv7dH7YbJ2gBHPqS7E6fQg41tszqmHbzmg?e=wyZOKW) and put it under `VirtualPose`.
  ```bash
  git clone https://github.com/wkom/VirtualPose.git
  cd VirtualPose
  python setup.py develop
  ```

2. **Render Env.** If you run this code in ssh environment without display device, please do follow:
  ```
  1. Install osmesa follow https://pyrender.readthedocs.io/en/latest/install/
  2. Reinstall the specific pyopengl fork: https://github.com/mmatl/pyopengl
  3. Set opengl's backend to osmesa via os.environ["PYOPENGL_PLATFORM"] = "osmesa"
  ```

3. **Model weight.** Download the pre-trained VMarker-Pro models `baseline` from [Onedrive](https://chinapku-my.sharepoint.com/:u:/g/personal/2101111546_pku_edu_cn/ESX2io93u95BqEsldgjc520Bq5k77CWmzFO32furG7rhzA?e=dP2Vco). Put the weight below `experiment` folder and follow the directory structure. Specify the load weight path by `test.weight_path` in `configs/diff3dmesh_infer/baseline.yml`.

4. **Input image/video.** Prepare `input.jpg` or `input.mp4` and put it in `inputs` folder. Both image and video input are supported. Specify the input path and type by arguments.

5. **RUN.** You can check the output at `experiment/diff3dmesh_infer/exp_*/vis`. By default, we output only one solution. If multiple hypotheses are needed, please set `cfg.test.multi_n` accordingly.
  ```bash
  sh command/diff3dmesh_infer/baseline.sh
  ```

## Train & Eval

### Data

Please follow [VMarker](https://github.com/ShirleyMaxx/VirtualMarker) to prepare the `data` directory.


### Train

Every experiment is defined by `config` files. Configs of the experiments in the paper can be found in the `./configs` directory. You can use the scripts under `command` to run.

To train the model, simply run the script below. Specific configurations can be modified in the corresponding `configs/diff3dmesh_train/baseline.yml` file. Default setting is using 2 GPUs (80G A100). Multi-GPU training is implemented with PyTorch's [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). Results can be seen in `experiment` directory or in the tensorboard.

We conduct mix-training on H3.6M and 3DPW datasets. To get the reported results on 3DPW dataset, please first run `train_h36m.sh` and then load the final weight to train on 3DPW or SURREAL by running `train_pw3d.sh` or `train_surreal.sh`. 

```bash
sh command/diff3dmesh_train/train_h36m.sh
sh command/diff3dmesh_train/train_pw3d.sh
sh command/diff3dmesh_train/train_surreal.sh
```

### Evaluation

To evaluate the model, specify the model path `test.weight_path` in `configs/diff3dmesh_test/baseline_*.yml`. Argument `--mode test` should be set. Results can be seen in `experiment` directory or in the tensorboard.

```bash
sh command/diff3dmesh_test/test_h36m.sh
sh command/diff3dmesh_test/test_pw3d.sh
sh command/diff3dmesh_test/test_surreal.sh
```

### Model Zoo

Download all model weights from [Onedrive](https://chinapku-my.sharepoint.com/:f:/g/personal/2101111546_pku_edu_cn/EnBL5ZBwmeVLrbIZ2Id9wbgB0SYmK_aE4crCYf3nX5RoNw?e=ADy57b) and put them under `experiment` folder.


<!-- ## Citation
Cite below if you find this repository helpful to your project:
```bibtex
@article{ma20233d,
  title={VMarker-Pro: Probabilistic 3D Human Mesh Estimation from Virtual Markers},
  author={Ma, Xiaoxuan and Su, Jiajun and Xu, Yuan and Zhu, Wentao and Wang, Chunyu and Wang, Yizhou},
  journal={arXiv preprint},
  year={2024}
}
``` -->

## Acknowledgement
This repo is built on the excellent work [Pose2Mesh](https://github.com/hongsukchoi/Pose2Mesh_RELEASE), [HybrIK](https://github.com/Jeff-sjtu/HybrIK), [CLIFF](https://github.com/haofanwang/CLIFF) and [LDM](https://github.com/CompVis/latent-diffusion). Thanks for these great projects.
