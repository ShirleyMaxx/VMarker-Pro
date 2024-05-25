import os
import argparse
import torch
import shutil
import time
import numpy as np
import scipy.sparse as ssp
import os.path as osp
import logging
from vmpro.utils.funcs_utils import save_checkpoint, check_data_pararell
from vmpro.utils.ema_utils import EMAHelper
from vmpro.core.config import cfg, update_config, init_experiment_dir
from vmpro.core.function import Simple3DMeshTrainer, Simple3DMeshTester
from vmpro.core.function import Diffusion3DMeshTrainer, Diffusion3DMeshTester

def parse_args():
    parser = argparse.ArgumentParser(description='Train/test VMarker & VMarker-Pro')

    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--resume_training', action='store_true', help='Resume Training')
    parser.add_argument('--ddp', action='store_true', help='DDP Training if set')
    parser.add_argument('--gpus', type=int, default=4, help='number of gpus')
    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument('--sync_bn', action='store_true', help="If set, then utilize pytorch convert_syncbn_model")
    parser.add_argument('--cfg', type=str, help='experiment configure file name')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--experiment_name', type=str, default='', help='experiment name')
    parser.add_argument('--data_path', type=str, default='.', help='data dir path')
    parser.add_argument('--cur_path', type=str, default='.', help='current dir path')
    parser.add_argument('--debug_t', type=int, default=-1, help='debug timestep')

    args = parser.parse_args()
    return args

def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"
    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl")

    return True

def main(args):

    is_distributed = False
    is_distributed = init_distributed(args)
    args.is_distributed = is_distributed

    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
    else:
        rank = world_size = None
        logging.basicConfig(level=logging.INFO)
    args.rank = rank
    args.world_size = world_size

    if is_distributed:
        args.device = torch.device(args.local_rank)
    else:
        args.device = [i for i in range(args.gpus)]

    # experiment
    writer = None
    logging.info("Number of available GPUs: {}".format(torch.cuda.device_count()))
    logging.info("args: {}".format(args))
    if master:
        writer = init_experiment_dir(args.cur_path, args.data_path, args.experiment_name)
        shutil.copy(args.cfg, cfg.checkpoint_dir)
        shutil.copy('vmpro/models/diff3dmesh.py', cfg.checkpoint_dir)
    cfg.data_dir = osp.join(args.data_path, 'data')
    # update config
    if args.cfg:
        update_config(args.cfg)

    load_path_train = cfg.train.resume_weight_path
    load_path_test = cfg.test.weight_path if args.mode == 'test' else ''
    model_name = cfg.model.name

    cfg.diffusion.debug_timestep = int(args.debug_t)

    if model_name == 'simple3dmesh':
        trainer = Simple3DMeshTrainer(args, load_path=load_path_train, writer=writer, master=master) if args.mode == 'train' else None
        tester = Simple3DMeshTester(args, load_path=load_path_test, writer=writer, master=master)  # if not args.debug else None
    elif model_name == 'diff3dmesh':
        trainer = Diffusion3DMeshTrainer(args, load_path=load_path_train, writer=writer, master=master) if args.mode == 'train' else None
        tester = Diffusion3DMeshTester(args, load_path=load_path_test, writer=writer, master=master)  # if not args.debug else None


    logging.info(f"===> Start {args.mode}ing...")
    n_iters_total_train, n_iters_total_val = 0, 0
    for epoch in range(cfg.train.begin_epoch, cfg.train.end_epoch + 1):
        if args.mode == 'train':
            if trainer.sampler is not None:
                trainer.sampler.set_epoch(epoch)
            n_iters_total_train = trainer.train(epoch, n_iters_total_train, master)
            if model_name == 'simple3dmesh':
                if not cfg.model.simple3dmesh.noise_reduce:
                    trainer.lr_scheduler['3d'].step()
                    trainer.lr_scheduler['mesh'].step()
                else:
                    trainer.lr_scheduler['nr'].step()
            elif model_name == 'diff3dmesh':
                if not cfg.model.simple3dpose.fix_network:
                    trainer.lr_scheduler['3d'].step()
                if not cfg.model.simple3dmesh.fix_network:
                    trainer.lr_scheduler['mesh'].step()   
                trainer.lr_scheduler['diff'].step()   

        if args.mode == 'test':
            tester.test(0, master, world_size)
            break

        tester.test(epoch, master, world_size, current_model=trainer.model)
        
        is_best = None

        if master:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': check_data_pararell(trainer.model.state_dict()),
                'train_log': trainer.loss_history,
            }
            if model_name == 'diff3dmesh' and cfg.diffusion.ema:
                save_dict.update({'ema_helper': trainer.ema_helper.state_dict()})
            if model_name == 'simple3dmesh':
                save_dict.update({
                    'optim_state_dict_3d': trainer.optimizer['3d'].state_dict(),
                    'scheduler_state_dict_3d': trainer.lr_scheduler['3d'].state_dict(),
                })

                if not cfg.model.simple3dmesh.fix_network:
                    save_dict.update({
                        'optim_state_dict_mesh': trainer.optimizer['mesh'].state_dict(),
                        'scheduler_state_dict_mesh': trainer.lr_scheduler['mesh'].state_dict(),
                    })
            elif model_name == 'diff3dmesh':
                if not cfg.model.simple3dpose.fix_network:
                    save_dict.update({
                        'optim_state_dict_3d': trainer.optimizer['3d'].state_dict(),
                        'scheduler_state_dict_3d': trainer.lr_scheduler['3d'].state_dict(),
                    })
                if not cfg.model.simple3dmesh.fix_network:
                    save_dict.update({
                        'optim_state_dict_mesh': trainer.optimizer['mesh'].state_dict(),
                        'scheduler_state_dict_mesh': trainer.lr_scheduler['mesh'].state_dict(),
                    })
                save_dict.update({
                    'optim_state_dict_diff': trainer.optimizer['diff'].state_dict(),
                    'scheduler_state_dict_diff': trainer.lr_scheduler['diff'].state_dict(),
                })
            else:
                save_dict.update({
                    'optim_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
                })            
            save_checkpoint(save_dict, epoch, is_best)


if __name__ == '__main__':
    args = parse_args()
    if args.ddp:
        main(args)
    else:
        if (int(os.environ.get("LOCAL_RANK", 0)) == 0):
            main(args)
