# ft on SURREAL
python -m torch.distributed.launch --nproc_per_node=4 --master_port=20349 main/main.py --cfg ./configs/simple3dmesh_train/baseline_surreal.yml --experiment_name simple3dmesh_train/baseline_surreal --gpus 4 --ddp --sync_bn --resume_training