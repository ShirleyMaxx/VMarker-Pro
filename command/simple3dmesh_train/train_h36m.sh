# first, train for H36M
python -m torch.distributed.launch --nproc_per_node=2 --master_port=20349 main/main.py --cfg ./configs/simple3dmesh_train/baseline_h36m.yml --experiment_name simple3dmesh_train/baseline_h36m --gpus 2 --ddp --sync_bn

# # second, train on PW3D training set for PW3D
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=20349 main/main.py --cfg ./configs/simple3dmesh_train/baseline_pw3d.yml --experiment_name simple3dmesh_train/baseline_pw3d --gpus 2 --ddp --sync_bn --resume_training
