python -m torch.distributed.launch --nproc_per_node=2 --master_port=19187 main/main.py --cfg ./configs/diff3dmesh_test/baseline_pw3d.yml --experiment_name diff3dmesh_test/baseline_pw3d --gpus 2 --ddp --mode test

# Test on 3DPW-OC
python -m torch.distributed.launch --nproc_per_node=2 --master_port=19187 main/main.py --cfg ./configs/diff3dmesh_test/baseline_pw3d_oc.yml --experiment_name diff3dmesh_test/baseline_pw3d_oc --gpus 2 --ddp --mode test

# Test on 3DPW-PC
python -m torch.distributed.launch --nproc_per_node=2 --master_port=19187 main/main.py --cfg ./configs/diff3dmesh_test/baseline_pw3d_pc.yml --experiment_name diff3dmesh_test/baseline_pw3d_pc --gpus 2 --ddp --mode test
