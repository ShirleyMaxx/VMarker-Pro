# ft on 3DPW
python -m torch.distributed.launch --nproc_per_node=2 --master_port=20349 main/main.py --cfg ./configs/diff3dmesh_train/baseline_pw3d.yml --experiment_name diff3dmesh_train/baseline_pw3d --gpus 2 --ddp --sync_bn --resume_training