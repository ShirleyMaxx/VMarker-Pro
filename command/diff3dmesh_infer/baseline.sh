# add ``--test_fps'' to test fps 
CUDA_VISIBLE_DEVICES=0 python main/inference_pro.py --cfg ./configs/diff3dmesh_infer/baseline.yml --experiment_name diff3dmesh_infer/baseline --gpu 1 --input_path inputs/input.mp4 --input_type video --save_vid 
