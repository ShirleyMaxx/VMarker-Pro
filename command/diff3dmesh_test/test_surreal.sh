python -m torch.distributed.launch --nproc_per_node=2 --master_port=19187 main/main.py --cfg ./configs/diff3dmesh_test/baseline_surreal.yml --experiment_name diff3dmesh_test/baseline_surreal --gpus 2 --ddp --mode test