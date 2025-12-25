import os
import glob
import logging
import importlib
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.prefetch_dataloader import PrefetchDataLoader, CPUPrefetcher
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torch.utils.tensorboard import SummaryWriter

from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from core.loss import AdversarialLoss, PerceptualLoss, LPIPSLoss
from core.dataset import TrainDataset

from model.modules.flow_comp_raft import RAFT_bi, FlowLoss, EdgeLoss
from model.recurrent_flow_completion import RecurrentFlowCompleteNet

from RAFT.utils.flow_viz_pt import flow_to_image


class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0

        # =========================================================
        # [Compatibility Layer] 自动适配层
        # =========================================================
        if 'trainer' not in self.config: self.config['trainer'] = {}
        if 'model' not in self.config: self.config['model'] = {}

        net_name = 'propainter'
        if 'network_g' in self.config:
            net_type = self.config['network_g'].get('type', 'propainter')
            net_name = 'propainter' if net_type == 'ProPainter' else net_type

        if 'net' not in self.config['model']: self.config['model']['net'] = net_name
        if 'interp_mode' not in self.config['model']: self.config['model']['interp_mode'] = 'nearest'
        if 'losses' not in self.config: self.config['losses'] = {}

        if 'train' in self.config:
            train_opt = self.config['train']
            if 'optim_g' in train_opt:
                self.config['trainer']['lr'] = train_opt['optim_g'].get('lr', 1e-4)
                self.config['trainer']['beta1'] = train_opt['optim_g'].get('betas', [0.9, 0.99])[0]
                self.config['trainer']['beta2'] = train_opt['optim_g'].get('betas', [0.9, 0.99])[1]
            else:
                self.config['trainer'].setdefault('lr', 1e-4)
                self.config['trainer'].setdefault('beta1', 0.9)
                self.config['trainer'].setdefault('beta2', 0.99)

            if 'scheduler' not in self.config['trainer']: self.config['trainer']['scheduler'] = {}
            if 'scheduler' in train_opt:
                sch_opt = train_opt['scheduler']
                sch_type = sch_opt.get('type', 'MultiStepLR')
                if sch_type == 'MultiStepLR':
                    self.config['trainer']['scheduler']['type'] = 'MultiStepRestartLR'
                    self.config['trainer']['scheduler']['milestones'] = sch_opt.get('milestones', [400000])
                    self.config['trainer']['scheduler']['gamma'] = sch_opt.get('gamma', 0.5)
                elif sch_type == 'CosineAnnealingRestartLR':
                    self.config['trainer']['scheduler']['type'] = 'CosineAnnealingRestartLR'
                    self.config['trainer']['scheduler']['periods'] = sch_opt.get('periods', [150000])
                    self.config['trainer']['scheduler']['restart_weights'] = sch_opt.get('restart_weights', [1])
                    self.config['trainer']['scheduler']['eta_min'] = sch_opt.get('eta_min', 1e-7)
                else:
                    self.config['trainer']['scheduler']['type'] = 'MultiStepRestartLR'
                    self.config['trainer']['scheduler']['milestones'] = [400000]
                    self.config['trainer']['scheduler']['gamma'] = 0.5
            else:
                self.config['trainer']['scheduler'].update(
                    {'type': 'MultiStepRestartLR', 'milestones': [400000], 'gamma': 0.5})

            if 'gan_opt' in train_opt:
                self.config['losses']['GAN_LOSS'] = train_opt['gan_opt'].get('type', 'hinge')
                self.config['losses']['adversarial_weight'] = train_opt['gan_opt'].get('loss_weight', 0.01)
            else:
                self.config['losses'].update({'GAN_LOSS': 'hinge', 'adversarial_weight': 0.01})

            pixel_weight = train_opt['pixel_opt'].get('loss_weight', 1.0) if 'pixel_opt' in train_opt else 1.0
            self.config['losses']['hole_weight'] = pixel_weight
            self.config['losses']['valid_weight'] = pixel_weight

            perc_weight = train_opt['perceptual_opt'].get('perceptual_weight',
                                                          1.0) if 'perceptual_opt' in train_opt else 0.0
            self.config['losses']['perceptual_weight'] = perc_weight
        else:
            self.config['trainer'].setdefault('lr', 1e-4)
            self.config['trainer'].setdefault('beta1', 0.9)
            self.config['trainer'].setdefault('beta2', 0.99)
            self.config['trainer']['scheduler'] = {'type': 'MultiStepRestartLR', 'milestones': [400000], 'gamma': 0.5}
            self.config['losses'].update(
                {'GAN_LOSS': 'hinge', 'adversarial_weight': 0.01, 'hole_weight': 1.0, 'valid_weight': 1.0,
                 'perceptual_weight': 1.0})

        if 'no_dis' not in self.config['model']:
            self.config['model']['no_dis'] = True if self.config['losses']['adversarial_weight'] <= 0 else False

        if 'iterations' not in self.config['trainer']:
            self.config['trainer']['iterations'] = self.config['train'][
                'total_iter'] if 'train' in self.config and 'total_iter' in self.config['train'] else 400000
        if 'log_freq' not in self.config['trainer']:
            self.config['trainer']['log_freq'] = self.config['logger'][
                'print_freq'] if 'logger' in self.config and 'print_freq' in self.config['logger'] else 100
        if 'save_freq' not in self.config['trainer']:
            self.config['trainer']['save_freq'] = self.config['logger'][
                'save_checkpoint_freq'] if 'logger' in self.config and 'save_checkpoint_freq' in self.config[
                'logger'] else 5000

        if 'train_data_loader' in config:
            self.num_local_frames = config['train_data_loader']['num_local_frames']
            self.num_ref_frames = config['train_data_loader']['num_ref_frames']
            data_loader_config = config['train_data_loader']
        else:
            self.num_local_frames = config['datasets']['train'].get('num_frame', 5)
            self.num_ref_frames = config['datasets']['train'].get('num_ref_frames', 0)
            data_loader_config = config['datasets']['train']

        self.train_dataset = TrainDataset(data_loader_config)
        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=config['world_size'],
                                                    rank=config['global_rank'])

        if 'batch_size' in self.train_args:
            local_batch_size = self.train_args['batch_size'] // config['world_size']
            local_num_workers = self.train_args.get('num_workers', 4)
        else:
            train_opt = config['datasets']['train']
            local_batch_size = train_opt.get('batch_size_per_gpu', 4)
            local_num_workers = train_opt.get('num_worker_per_gpu', 4)

        dataloader_args = dict(dataset=self.train_dataset, batch_size=local_batch_size,
                               shuffle=(self.train_sampler is None), num_workers=local_num_workers,
                               sampler=self.train_sampler, drop_last=True)
        self.train_loader = PrefetchDataLoader(self.train_args.get('num_prefetch_queue', 1), **dataloader_args)
        self.prefetcher = CPUPrefetcher(self.train_loader)

        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS']).to(self.config['device'])
        self.l1_loss = nn.L1Loss()
        if self.config['losses']['perceptual_weight'] > 0:
            self.perc_loss = LPIPSLoss(use_input_norm=True, range_norm=True).to(self.config['device'])

        self.fix_raft = RAFT_bi(device=self.config['device'])
        self.fix_flow_complete = RecurrentFlowCompleteNet('weights/recurrent_flow_completion.pth')
        for p in self.fix_flow_complete.parameters(): p.requires_grad = False
        self.fix_flow_complete.to(self.config['device']).eval()

        net = importlib.import_module('model.' + config['model']['net'])
        self.netG = net.InpaintGenerator().to(self.config['device'])
        if not self.config['model'].get('no_dis', False):
            if self.config['model'].get('dis_2d', False):
                self.netD = net.Discriminator_2D(in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            else:
                self.netD = net.Discriminator(in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            self.netD = self.netD.to(self.config['device'])

        self.interp_mode = self.config['model']['interp_mode']
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        if config['distributed']:
            self.netG = DDP(self.netG, device_ids=[self.config['local_rank']], output_device=self.config['local_rank'],
                            broadcast_buffers=True, find_unused_parameters=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(self.netD, device_ids=[self.config['local_rank']],
                                output_device=self.config['local_rank'], broadcast_buffers=True,
                                find_unused_parameters=False)

        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            if not self.config['model']['no_dis']: self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(os.path.join(config['save_dir'], 'gen'))

    def setup_optimizers(self):
        backbone_params = []
        for name, param in self.netG.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
            else:
                print(f'Params {name} will not be optimized.')
        optim_params = [{'params': backbone_params, 'lr': self.config['trainer']['lr']}]
        self.optimG = torch.optim.Adam(optim_params,
                                       betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(self.netD.parameters(), lr=self.config['trainer']['lr'],
                                           betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))

    def setup_schedulers(self):
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(self.optimG, milestones=scheduler_opt['milestones'],
                                            gamma=scheduler_opt['gamma'])
            if not self.config['model']['no_dis']: self.scheD = MultiStepRestartLR(self.optimD,
                                                                                   milestones=scheduler_opt[
                                                                                       'milestones'],
                                                                                   gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(self.optimG, periods=scheduler_opt['periods'],
                                                  restart_weights=scheduler_opt['restart_weights'],
                                                  eta_min=scheduler_opt['eta_min'])
            if not self.config['model']['no_dis']: self.scheD = CosineAnnealingRestartLR(self.optimD,
                                                                                         periods=scheduler_opt[
                                                                                             'periods'],
                                                                                         restart_weights=scheduler_opt[
                                                                                             'restart_weights'],
                                                                                         eta_min=scheduler_opt[
                                                                                             'eta_min'])
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        self.scheG.step()
        if not self.config['model']['no_dis']: self.scheD.step()

    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

    def add_summary(self, writer, name, val):
        if name not in self.summary: self.summary[name] = 0
        self.summary[name] += val
        n = self.train_args['log_freq']
        if writer is not None and self.iteration % n == 0:
            writer.add_scalar(name, self.summary[name] / n, self.iteration)
            self.summary[name] = 0

    def load(self):
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1][4:] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path, f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path, f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path, f'opt_{int(latest_epoch):06d}.pth')
            if self.config['global_rank'] == 0: print(f'Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)
            if not self.config['model']['no_dis'] and self.config['model']['load_d']:
                dataD = torch.load(dis_path, map_location=self.config['device'])
                self.netD.load_state_dict(dataD)
            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])
            if not self.config['model']['no_dis'] and self.config['model']['load_d']:
                self.optimD.load_state_dict(data_opt['optimD'])
            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']
        else:
            gen_path = self.config['trainer'].get('gen_path', None)
            if gen_path is None and 'path' in self.config:
                gen_path = self.config['path'].get('pretrain_network_g', None)
                if self.config['global_rank'] == 0 and gen_path: print(f'[自动适配] 发现预训练权重配置: {gen_path}')

            dis_path = self.config['trainer'].get('dis_path', None)
            opt_path = self.config['trainer'].get('opt_path', None)
            if gen_path is not None:
                if self.config['global_rank'] == 0: print(f'Loading Gen-Net from {gen_path}...')
                dataG = torch.load(gen_path, map_location=self.config['device'])
                self.netG.load_state_dict(dataG)
                if dis_path is not None and not self.config['model']['no_dis'] and self.config['model']['load_d']:
                    if self.config['global_rank'] == 0: print(f'Loading Dis-Net from {dis_path}...')
                    dataD = torch.load(dis_path, map_location=self.config['device'])
                    self.netD.load_state_dict(dataD)
                if opt_path is not None:
                    data_opt = torch.load(opt_path, map_location=self.config['device'])
                    self.optimG.load_state_dict(data_opt['optimG'])
                    self.scheG.load_state_dict(data_opt['scheG'])
                    if not self.config['model']['no_dis'] and self.config['model']['load_d']:
                        self.optimD.load_state_dict(data_opt['optimD'])
                        self.scheD.load_state_dict(data_opt['scheD'])
            else:
                if self.config['global_rank'] == 0: print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

    def save(self, it):
        if self.config['global_rank'] == 0:
            gen_path = os.path.join(self.config['save_dir'], f'gen_{it:06d}.pth')
            dis_path = os.path.join(self.config['save_dir'], f'dis_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'], f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module if not self.config['model']['no_dis'] else None
            else:
                netG = self.netG
                netD = self.netD if not self.config['model']['no_dis'] else None

            torch.save(netG.state_dict(), gen_path)
            if not self.config['model']['no_dis']:
                torch.save(netD.state_dict(), dis_path)
                torch.save({'epoch': self.epoch, 'iteration': self.iteration, 'optimG': self.optimG.state_dict(),
                            'optimD': self.optimD.state_dict(), 'scheG': self.scheG.state_dict(),
                            'scheD': self.scheD.state_dict()}, opt_path)
            else:
                torch.save({'epoch': self.epoch, 'iteration': self.iteration, 'optimG': self.optimG.state_dict(),
                            'scheG': self.scheG.state_dict()}, opt_path)
            os.system(f"echo {it:06d} > {os.path.join(self.config['save_dir'], 'latest.ckpt')}")

    def train(self):
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0: pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True,
                                                        smoothing=0.01)
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s %(message)s",
                            datefmt="%a, %d %b %Y %H:%M:%S",
                            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log", filemode='w')
        while True:
            self.epoch += 1
            self.prefetcher.reset()
            if self.config['distributed']: self.train_sampler.set_epoch(self.epoch)
            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']: break
        print('\nEnd training....')

    def _train_epoch(self, pbar):
        device = self.config['device']
        train_data = self.prefetcher.next()
        while train_data is not None:
            self.iteration += 1
            if isinstance(train_data, dict):
                frames = train_data['lq'].to(device)
                gt_frames = train_data['gt'].to(device)
                masks = train_data['mask'].to(device).float()
                flows_f = train_data.get('flow_f', ['None'])
                flows_b = train_data.get('flow_b', ['None'])
                alpha_map = train_data['alpha'].to(device).float() if 'alpha' in train_data else None
            else:
                frames, masks, flows_f, flows_b, _ = train_data
                frames, masks = frames.to(device), masks.to(device).float()
                gt_frames = frames
                alpha_map = None

            l_t = self.num_local_frames
            b, t, c, h, w = frames.size()
            gt_local_frames = gt_frames[:, :l_t, ...]
            local_masks = masks[:, :l_t, ...].contiguous()

            # [核心修改: 双流输入策略]
            # Input A: 挖空图 -> 专门给 RAFT 光流看，强迫其寻找背景运动
            # gt_frames 是干净的，所以我们人工造一个挖空图
            masked_frames_for_flow = gt_frames * (1 - masks)

            # Input B: 原始脏图 -> 专门给 ProPainter 看，让其利用残余像素反解水印
            # frames 本身就是 input (带水印的)
            raw_dirty_frames = frames

            # 切片
            masked_local_for_flow = masked_frames_for_flow[:, :l_t, ...]
            masked_local_frames = raw_dirty_frames[:, :l_t, ...]  # 脏图切片

            # 1. 计算光流 (强制使用挖空图!)
            if isinstance(flows_f, list) and (flows_f[0] == 'None' or flows_b[0] == 'None'):
                # RAFT 看到的是黑洞，所以它会尽力去匹配周围的背景
                gt_flows_bi = self.fix_raft(masked_local_for_flow)
            else:
                gt_flows_bi = (flows_f.to(device), flows_b.to(device))

            pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, local_masks)
            pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, local_masks)

            # [兼容性修复] 获取模型本体
            if isinstance(self.netG, (DDP, torch.nn.DataParallel)):
                netG_interface = self.netG.module
            else:
                netG_interface = self.netG

            # 2. 图像传播与修复 (强制使用原图!)
            # 我们把带水印的原图喂给 img_propagation，让它有机会利用半透明信息
            prop_imgs, updated_local_masks = netG_interface.img_propagation(
                masked_local_frames,  # <--- 传入脏图 (raw_dirty_frames)
                pred_flows_bi,
                local_masks,
                interpolation=self.interp_mode
            )

            updated_masks = masks.clone()
            updated_masks[:, :l_t, ...] = updated_local_masks.view(b, l_t, 1, h, w)

            # 3. 更新 Transformer 的输入 (基于脏图)
            updated_frames = raw_dirty_frames.clone()  # <--- 脏图底板
            prop_local_frames = gt_local_frames * (1 - local_masks) + prop_imgs.view(b, l_t, 3, h, w) * local_masks
            updated_frames[:, :l_t, ...] = prop_local_frames

            # Transformer 预测
            pred_imgs = self.netG(updated_frames, pred_flows_bi, masks, updated_masks, l_t)
            pred_imgs = pred_imgs.view(b, -1, c, h, w)
            # pred_local_frames = pred_imgs[:, :l_t, ...]
            pred_local_frames = pred_imgs  # 修正维度引用
            comp_imgs = gt_frames * (1. - masks) + pred_imgs * masks

            gen_loss = 0
            dis_loss = 0
            if not self.config['model']['no_dis']:
                for p in self.netD.parameters(): p.requires_grad = False
            self.optimG.zero_grad()

            # --- 计算各个 Loss ---
            # 1. Hole Loss (修复：加 1e-8 防止除零)
            hole_loss = self.l1_loss(pred_imgs * masks, gt_frames * masks)
            hole_loss = hole_loss / (torch.mean(masks) + 1e-8) * self.config['losses']['hole_weight']
            gen_loss += hole_loss
            self.add_summary(self.gen_writer, 'loss/hole_loss', hole_loss.item())

            # 2. Valid Loss (修复：加 1e-8 防止除零)
            valid_loss = self.l1_loss(pred_imgs * (1 - masks), gt_frames * (1 - masks))
            valid_loss = valid_loss / (torch.mean(1 - masks) + 1e-8) * self.config['losses']['valid_weight']
            gen_loss += valid_loss
            self.add_summary(self.gen_writer, 'loss/valid_loss', valid_loss.item())

            # 3. Perceptual Loss
            perc_loss = torch.tensor(0.0).to(device)
            if self.config['losses']['perceptual_weight'] > 0:
                perc_loss = self.perc_loss(pred_imgs.view(-1, 3, h, w), gt_frames.view(-1, 3, h, w))[0] * \
                            self.config['losses']['perceptual_weight']
                gen_loss += perc_loss
                self.add_summary(self.gen_writer, 'loss/perc_loss', perc_loss.item())

            # 4. Physics Loss (关键！！)
            physics_loss_val = 0.0
            physics_loss = torch.tensor(0.0).to(device)
            if alpha_map is not None:
                if alpha_map.dim() == 4: alpha_map = alpha_map.unsqueeze(1).expand_as(masks)
                if alpha_map.max() > 1.0: alpha_map = alpha_map / 255.0

                alpha_weight = alpha_map
                # 这里的 pred_imgs 是模型输出的去水印结果
                # gt_frames 是无水印真值
                # 物理 Loss 计算逻辑： |(pred - gt) * alpha|
                numerator = (torch.abs(pred_imgs - gt_frames) * alpha_weight).sum()
                denominator = alpha_weight.sum() + 1e-8

                physics_loss = numerator / denominator
                # 截断
                physics_loss = torch.clamp(physics_loss, min=0.0, max=10.0)

                # 必须从 config 读取，或者硬编码为高值
                lambda_physics = self.config['train'].get('lambda_physics', 20.0)

                gen_loss += lambda_physics * physics_loss
                physics_loss_val = physics_loss.item()
                self.add_summary(self.gen_writer, 'loss/physics_loss', physics_loss_val)

            # 5. GAN Loss
            gan_loss_val = 0.0
            if not self.config['model']['no_dis']:
                gen_clip = self.netD(comp_imgs)
                gan_loss = self.adversarial_loss(gen_clip, True, False) * self.config['losses']['adversarial_weight']
                gen_loss += gan_loss
                gan_loss_val = gan_loss.item()
                self.add_summary(self.gen_writer, 'loss/gan_loss', gan_loss_val)

            # --- NaN 诊断与梯度裁剪 ---
            if torch.isnan(gen_loss):
                print(f"\n[FATAL] Iter {self.iteration} NaN Detected! Breakdown:")
                print(f"  - Hole: {hole_loss.item()}")
                print(f"  - Valid: {valid_loss.item()}")
                print(f"  - Perc: {perc_loss.item()}")
                print(f"  - Phy: {physics_loss_val}")
                print(f"  - GAN: {gan_loss_val}")
                self.optimG.zero_grad()
            else:
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1.0)
                self.optimG.step()
                self.update_learning_rate()

                if not self.config['model']['no_dis']:
                    for p in self.netD.parameters(): p.requires_grad = True
                    self.optimD.zero_grad()
                    real_clip = self.netD(gt_frames)
                    fake_clip = self.netD(comp_imgs.detach())
                    dis_real_loss = self.adversarial_loss(real_clip, True, True)
                    dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                    dis_loss += (dis_real_loss + dis_fake_loss) / 2
                    self.add_summary(self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
                    self.add_summary(self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())

                    # 修复：检查 dis_loss 是否是 NaN
                    if isinstance(dis_loss, torch.Tensor) and not torch.isnan(dis_loss):
                        dis_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), 1.0)
                        self.optimD.step()

            # --- 可视化 ---
            if self.iteration % 200 == 0:
                t = 0
                try:
                    gt_cpu = ((gt_local_frames.view(b, -1, 3, h, w) + 1) / 2.0).cpu()
                    # masked_cpu 显示脏图
                    masked_cpu = ((masked_local_frames.view(b, -1, 3, h, w) + 1) / 2.0).cpu()
                    pred_cpu = ((pred_local_frames.view(b, -1, 3, h, w) + 1) / 2.0).cpu()
                    img_results = torch.cat([masked_cpu[0][t], gt_cpu[0][t], pred_cpu[0][t]], 1)
                    img_results = torchvision.utils.make_grid(img_results, nrow=1, normalize=True)
                    if self.gen_writer is not None:
                        self.gen_writer.add_image(f'img/compare_step_{self.iteration}', img_results, self.iteration)
                except Exception as e:
                    print(f"Warning: Visualization failed: {e}")

            if self.config['global_rank'] == 0:
                pbar.update(1)
                h_v = hole_loss.item() if not torch.isnan(hole_loss) else 0
                p_v = physics_loss_val if not math.isnan(physics_loss_val) else 0

                log_msg = f"hole: {h_v:.3f}; phy: {p_v:.3f}"
                if not self.config['model']['no_dis']:
                    d_v = dis_loss.item() if isinstance(dis_loss, torch.Tensor) and not torch.isnan(dis_loss) else 0
                    log_msg = f"d: {d_v:.3f}; " + log_msg

                pbar.set_description(log_msg)
                if self.iteration % self.train_args['log_freq'] == 0:
                    logging.info(f"[Iter {self.iteration}] " + log_msg)

            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))
            if self.iteration > self.train_args['iterations']: break
            train_data = self.prefetcher.next()