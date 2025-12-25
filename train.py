import os
import json
import argparse
import subprocess
import yaml
from shutil import copyfile
import torch.distributed as dist
import torch
import torch.multiprocessing as mp
import core
import core.trainer
# import core.trainer_flow_w_edge # 如果没用到可以注释掉

from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

parser = argparse.ArgumentParser()
parser.add_argument('-opt', '-c', '--config', default='configs/train_propainter.json', type=str)
parser.add_argument('-p', '--port', default='23490', type=str)
args = parser.parse_args()


def main_worker(rank, config):
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank
    if config['distributed']:
        torch.cuda.set_device(int(config['local_rank']))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=config['init_method'],
                                             world_size=config['world_size'],
                                             rank=config['global_rank'],
                                             group_name='mtorch')
        print('using GPU {}-{} for training'.format(int(config['global_rank']), int(config['local_rank'])))

    # [补丁1] 自动补全 save_dir
    if 'save_dir' not in config:
        if 'path' in config and 'experiments_root' in config['path']:
            config['save_dir'] = config['path']['experiments_root']
        else:
            config['save_dir'] = './experiments'

    # [补丁2] 智能获取模型名称
    model_name = 'ProPainter'
    if 'model' in config and 'net' in config['model']:
        model_name = config['model']['net']
    elif 'network_g' in config and 'type' in config['network_g']:
        model_name = config['network_g']['type']

    config['save_dir'] = os.path.join(config['save_dir'],
                                      '{}_{}'.format(model_name, os.path.basename(args.config).split('.')[0]))
    config['save_metric_dir'] = os.path.join('./scores',
                                             '{}_{}'.format(model_name, os.path.basename(args.config).split('.')[0]))

    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else:
        config['device'] = 'cpu'

    if (not config['distributed']) or config['global_rank'] == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        config_path = os.path.join(config['save_dir'], args.config.split('/')[-1])
        if not os.path.isfile(config_path):
            copyfile(args.config, config_path)
        print('[**] create folder {}'.format(config['save_dir']))

    # [补丁3] 自动补全 trainer 配置
    if 'trainer' not in config:
        config['trainer'] = {}
    if 'version' not in config['trainer']:
        config['trainer']['version'] = 'trainer'

    trainer_version = config['trainer']['version']
    trainer = core.__dict__[trainer_version].__dict__['Trainer'](config)
    trainer.train()


if __name__ == "__main__":
    # 强制让 print 立即输出，不缓存
    import sys
    import functools

    print = functools.partial(print, flush=True)

    print("DEBUG: 正在初始化...")

    torch.backends.cudnn.benchmark = True

    # 只有在多卡模式下才需要设置这个
    if torch.cuda.device_count() > 1:
        mp.set_sharing_strategy('file_system')

    # 支持 YAML 读取
    print(f"DEBUG: 正在读取配置文件 {args.config}...")
    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    else:
        config = json.load(open(args.config))

    # [修复] 强制使用 YAML 中配置的 GPU 数量
    target_gpu_num = config.get('num_gpu', 1)

    # 再次检查物理设备是否足够
    available_gpus = torch.cuda.device_count()
    if target_gpu_num > available_gpus:
        print(
            f"Warning: Config requests {target_gpu_num} GPUs but only {available_gpus} available. Using {available_gpus}.")
        target_gpu_num = available_gpus

    config['world_size'] = target_gpu_num
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"
    config['distributed'] = True if config['world_size'] > 1 else False

    print(f'world_size (configured): {config["world_size"]}')

    # ========================================================
    # [核心修改] 单卡直接运行，多卡才用 spawn
    # 这样可以解决单卡“假死”无输出的问题，并能看到所有报错
    # ========================================================
    if target_gpu_num == 1:
        print("DEBUG: 检测到单卡模式，直接启动 main_worker (非多进程模式)...")
        # 直接调用函数，报错会直接打印
        main_worker(0, config)
    else:
        print(f"DEBUG: 检测到多卡模式 ({target_gpu_num} GPUs)，启动多进程 spawn...")
        mp.spawn(main_worker, nprocs=target_gpu_num, args=(config,))