python -m torch.distributed.launch --nproc_per_node=4 --master_port=29495 main/main.py --cfg ./configs/simple3dmesh_test/baseline_h36m.yml --experiment_name simple3dmesh_test/baseline_h36m --gpus 4 --ddp --mode test
