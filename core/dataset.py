import os
import cv2
# [新增] 强制关闭 OpenCV 的多线程，防止与 PyTorch DataLoader 死锁
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import math
import torch

import random
import numpy as np
import torch.utils.data as data
from torchvision import transforms


# 工具函数：读取图像并调整大小/归一化
def read_img(path, size=None, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)

    if img is None:
        print(f"FATAL ERROR: Cannot read image from {path}")
        # 如果读取失败，这里返回全黑图，但控制台会报错
        if size is not None:
            h, w = size
            if grayscale:
                return np.zeros((h, w), dtype=np.uint8)
            else:
                return np.zeros((h, w, 3), dtype=np.uint8)
        return None

    if size is not None:
        img = cv2.resize(img, size)

    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


class VideoDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoDataset, self).__init__()
        self.opt = opt
        self.sample_stride = opt.get('sample_stride', 1)
        self.num_frame = opt.get('num_frame', 1)

        self.gt_root = opt['dataroot_gt']
        self.img_root = opt['dataroot_img']
        self.mask_root = opt['dataroot_mask']
        self.alpha_root = opt.get('dataroot_alpha', None)

        print(f"DEBUG: Dataset Init - GT Root: {self.gt_root}")

        self.video_names = sorted(os.listdir(self.gt_root))
        print(f"DEBUG: Found {len(self.video_names)} videos: {self.video_names}")

        self.img_paths = []
        self.gt_paths = []
        self.mask_paths = []
        self.alpha_paths = []

        for v_name in self.video_names:
            vid_gt_root = os.path.join(self.gt_root, v_name)
            vid_img_root = os.path.join(self.img_root, v_name)
            vid_mask_root = os.path.join(self.mask_root, v_name)

            vid_alpha_root = None
            if self.alpha_root is not None:
                vid_alpha_root = os.path.join(self.alpha_root, v_name)

            # 检查文件夹是否存在
            if not os.path.isdir(vid_gt_root):
                print(f"WARNING: {vid_gt_root} is not a directory, skipping.")
                continue

            frame_names = sorted(os.listdir(vid_gt_root))

            v_img_paths = []
            v_gt_paths = []
            v_mask_paths = []
            v_alpha_paths = []

            for f_name in frame_names:
                v_gt_paths.append(os.path.join(vid_gt_root, f_name))
                v_img_paths.append(os.path.join(vid_img_root, f_name))
                v_mask_paths.append(os.path.join(vid_mask_root, f_name))
                if vid_alpha_root is not None:
                    v_alpha_paths.append(os.path.join(vid_alpha_root, f_name))

            self.img_paths.append(v_img_paths)
            self.gt_paths.append(v_gt_paths)
            self.mask_paths.append(v_mask_paths)
            if vid_alpha_root is not None:
                self.alpha_paths.append(v_alpha_paths)

    def __getitem__(self, index):
        # 移除所有 print 以提升 IO 速度

        video_index = index % len(self.video_names)

        # [关键检查] 防止空文件夹导致除以0错误
        if len(self.gt_paths[video_index]) == 0:
            raise ValueError(f"FATAL ERROR: Video {self.video_names[video_index]} contains NO frames!")

        video_len = len(self.gt_paths[video_index])
        sample_len = self.num_frame * self.sample_stride

        if video_len >= sample_len:
            start_idx = random.randint(0, video_len - sample_len)
        else:
            start_idx = 0

        img_lq_list = []
        img_gt_list = []
        img_mask_list = []
        img_alpha_list = []

        for i in range(self.num_frame):
            current_idx = (start_idx + i * self.sample_stride) % video_len

            gt_path = self.gt_paths[video_index][current_idx]
            img_path = self.img_paths[video_index][current_idx]
            mask_path = self.mask_paths[video_index][current_idx]

            # 读取 GT
            img_gt = read_img(gt_path)

            # [强力纠错]
            if img_gt is None:
                raise FileNotFoundError(f"CRITICAL: Failed to read GT: {gt_path}")

            # 读取 Input
            img_lq = read_img(img_path, size=(img_gt.shape[1], img_gt.shape[0]))
            # 读取 Mask
            img_mask = read_img(mask_path, size=(img_gt.shape[1], img_gt.shape[0]), grayscale=True)

            if self.alpha_root is not None:
                alpha_path = self.alpha_paths[video_index][current_idx]
                img_alpha = read_img(alpha_path, size=(img_gt.shape[1], img_gt.shape[0]), grayscale=True)
            else:
                img_alpha = np.zeros_like(img_mask)

            # 归一化 [Important Fix]
            # 图片数据必须归一化到 [-1, 1] 以匹配 RAFT 和 Pretrained Model
            img_lq = (img_lq.astype(np.float32) / 255.0) * 2 - 1
            img_gt = (img_gt.astype(np.float32) / 255.0) * 2 - 1
            img_mask = img_mask.astype(np.float32) / 255.0
            img_alpha = img_alpha.astype(np.float32) / 255.0

            img_lq = torch.from_numpy(img_lq).permute(2, 0, 1)
            img_gt = torch.from_numpy(img_gt).permute(2, 0, 1)
            img_mask = torch.from_numpy(img_mask).unsqueeze(0)
            img_alpha = torch.from_numpy(img_alpha).unsqueeze(0)

            img_lq_list.append(img_lq)
            img_gt_list.append(img_gt)
            img_mask_list.append(img_mask)
            img_alpha_list.append(img_alpha)

        img_lqs = torch.stack(img_lq_list, dim=0)
        img_gts = torch.stack(img_gt_list, dim=0)
        img_masks = torch.stack(img_mask_list, dim=0)
        img_alphas = torch.stack(img_alpha_list, dim=0)

        gt_size = self.opt.get('gt_size', [432, 768])
        _, c, h, w = img_gts.size()
        target_h, target_w = gt_size

        if h > target_h and w > target_w:
            top = random.randint(0, h - target_h)
            left = random.randint(0, w - target_w)
            img_lqs = img_lqs[:, :, top:top + target_h, left:left + target_w]
            img_gts = img_gts[:, :, top:top + target_h, left:left + target_w]
            img_masks = img_masks[:, :, top:top + target_h, left:left + target_w]
            img_alphas = img_alphas[:, :, top:top + target_h, left:left + target_w]

        return {
            'lq': img_lqs,
            'gt': img_gts,
            'mask': img_masks,
            'alpha': img_alphas,
            'video_name': self.video_names[video_index]
        }

    def __len__(self):
        return len(self.video_names)


TrainDataset = VideoDataset