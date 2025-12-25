import os
import cv2
import numpy as np
import random
import glob
import math
import string
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

# ================= 配置区域 =================
SOURCE_DIR = "../datasets/source_videos"
FONT_DIR = "../datasets/fonts"  # 【重要】建议放入 Noto Sans CJK (思源黑体) 以支持多语言
OUTPUT_ROOT = "../datasets/result_videos/academic_dataset"

CLIP_DURATION = 4  # 切片时长
TARGET_FPS = 24  # 目标帧率
TARGET_SIZE = (1280, 720)  # 训练分辨率
NUM_THREADS = 8  # 并发线程
TRAIN_RATIO = 0.95  # 训练集比例

TEXT_CORPUS = [
    # --- 英文 (Latin) ---
    "CONFIDENTIAL", "DRAFT", "SAMPLE", "INTERNAL USE ONLY", "DO NOT COPY",
    "Evaluation Copy", "Created by ProPainter", "Top Secret", "Urgent",
    "Uploaded by User", "bilibili", "TikTok", "@Username", "Rec 00:01:23",
    "News Ticker", "Breaking News", "Live Broadcast", "HD 1080p", "4K HDR",
    "Trial Version", "Demo", "Copyright 2024", "All Rights Reserved",

    # --- 中文 (Simplified/Traditional Chinese) ---
    "绝密", "内部资料", "严禁外传", "仅供参考", "样本", "草稿",
    "测试视频", "高清", "蓝光", "独家", "首发", "直播回放",
    "禁止录屏", "版权所有", "盗版必究", "澳门首家线上赌场", "招商广告",
    "关注公众号", "加微信", "本台记者报道", "今日关注", "影像资料",

    # --- 日文 (Japanese - 混合假名与汉字) ---
    "見本", "複写禁止", "極秘", "速報", "生放送", "無断転載禁止",
    "サンプル", "コピー厳禁", "トップシークレット", "画質調整中",

    # --- 韩语 (Korean - 几何特征强) ---
    "견본", "복사 금지", "보안", "생방송", "녹화 중", "샘플",
    "저작권 소유", "배포 금지", "기밀", "테스트",

    # --- 俄语 (Cyrillic - 方块感强) ---
    "Образец", "Черновик", "Секретно", "Копия запрещена",
    "Прямой эфир", "Новости", "Архив", "Конфиденциально",

    # --- 混合/特殊字符 ---
    "© 2025 Studio", "VIP 会员专享", "ID: 9527", "UID: 88888888",
    "【转载】", "【搬运】", "www.website.com"
]


# ===========================================

def srgb_to_linear(img):
    img_norm = img.astype(np.float32) / 255.0
    return np.where(img_norm <= 0.04045, img_norm / 12.92, ((img_norm + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img):
    img = np.clip(img, 0, 1)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * (img ** (1 / 2.4)) - 0.055) * 255.0


# === 动态水印工厂 ===
class DynamicWatermarkFactory:
    def __init__(self, font_dir):
        # 扫描目录下所有字体
        self.fonts = glob.glob(os.path.join(font_dir, "*.[ot]tf"))
        if not self.fonts:
            print(f"[Warning] No fonts found in {font_dir}. Will use default PIL font (English Only!).")

    def generate_random_text_image(self):
        """随机生成一个带有文字的透明底板图片 (修复坐标偏移版)"""
        # 1. 随机选择文本
        if random.random() < 0.2:
            # 20% 概率生成随机乱码
            if random.random() < 0.5:
                length = random.randint(5, 15)
                text = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            else:
                length = random.randint(2, 5)
                text = "".join([chr(random.randint(0x4e00, 0x9fa5)) for _ in range(length)])
                text += str(random.randint(0, 99))
        else:
            text = random.choice(TEXT_CORPUS)

        # 2. 随机选择字体
        font_size = random.randint(60, 180)
        font = None
        if self.fonts:
            try:
                font_path = random.choice(self.fonts)
                font = ImageFont.truetype(font_path, font_size)
            except:
                pass

        if font is None:
            font = ImageFont.load_default()

        # 3. 随机颜色
        if random.random() < 0.7:
            val = random.randint(200, 255)
            r = g = b = val
        else:
            r, g, b = random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)

        base_alpha = random.randint(120, 255)

        # 4. === [核心修复] 精确计算画布尺寸与偏移 ===
        dummy_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))

        # 获取精确的边界框 (left, top, right, bottom)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 计算偏移量：为了让文字的左上角精确落在 (margin, margin) 处
        margin = 30
        x_offset = -bbox[0]
        y_offset = -bbox[1]

        # 画布总大小 = 文字实际大小 + 双倍边距
        canvas_w = text_width + margin * 2
        canvas_h = text_height + margin * 2

        img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # 5. 确定绘制坐标
        draw_x = margin + x_offset
        draw_y = margin + y_offset

        style = random.choice(['normal', 'outline', 'shadow', 'box'])

        if style == 'shadow':
            # 阴影也应用同样的偏移
            draw.text((draw_x + 4, draw_y + 4), text, font=font, fill=(0, 0, 0, base_alpha // 2))
            draw.text((draw_x, draw_y), text, font=font, fill=(r, g, b, base_alpha))

        elif style == 'outline':
            stroke_w = random.randint(2, 5)
            draw.text((draw_x, draw_y), text, font=font, fill=(r, g, b, base_alpha),
                      stroke_width=stroke_w, stroke_fill=(0, 0, 0, base_alpha))

        elif style == 'box':
            # 背景框逻辑
            box_pad_x = 10
            box_pad_y = 5
            rect_x1 = margin - box_pad_x
            rect_y1 = margin - box_pad_y
            rect_x2 = margin + text_width + box_pad_x
            rect_y2 = margin + text_height + box_pad_y

            draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2],
                           fill=(0, 0, 0, random.randint(100, 200)))
            draw.text((draw_x, draw_y), text, font=font, fill=(255, 255, 255, base_alpha))

        else:
            draw.text((draw_x, draw_y), text, font=font, fill=(r, g, b, base_alpha))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


# === 轨迹算法 ===
def get_bezier_curve(p0, p1, p2, p3, num_points):
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i in range(num_points):
        ct = t[i]
        curve[i] = (1 - ct) ** 3 * np.array(p0) + \
                   3 * (1 - ct) ** 2 * ct * np.array(p1) + \
                   3 * (1 - ct) * ct ** 2 * np.array(p2) + \
                   ct ** 3 * np.array(p3)
    return curve


def get_marquee_trajectory(start_x, y, end_x, num_points):
    """跑马灯直线轨迹"""
    xs = np.linspace(start_x, end_x, num_points)
    ys = np.full(num_points, y)
    return np.stack([xs, ys], axis=1)


def random_perspective_transform(img, strength=0.1):
    h, w = img.shape[:2]
    if h == 0 or w == 0: return img
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # 目标点 (随机偏移)
    dx, dy = w * strength, h * strength
    dst_pts = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],
        [w - random.uniform(0, dx), random.uniform(0, dy)],
        [random.uniform(0, dx), h - random.uniform(0, dy)],
        [w - random.uniform(0, dx), h - random.uniform(0, dy)]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


# === 水印对象 ===
class WatermarkObject:
    def __init__(self, factory, video_w, video_h, total_frames):
        # 1. 动态生成图片
        self.raw_img = factory.generate_random_text_image()

        # 2. 随机缩放
        h_raw, w_raw = self.raw_img.shape[:2]
        max_w_ratio = random.uniform(0.2, 0.7)
        target_w = int(video_w * max_w_ratio)
        scale = target_w / w_raw

        new_w, new_h = int(w_raw * scale), int(h_raw * scale)
        if new_w <= 0 or new_h <= 0: return

        self.img = cv2.resize(self.raw_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 3. 随机透视
        if random.random() < 0.3:
            self.img = random_perspective_transform(self.img, strength=0.08)

        self.h, self.w = self.img.shape[:2]
        self.img_rgb_linear = srgb_to_linear(self.img[:, :, :3])
        self.img_alpha_base = self.img[:, :, 3] / 255.0

        # 4. 运动模式决策
        mode = random.choices(['static', 'float', 'marquee'], weights=[0.4, 0.4, 0.2])[0]

        if mode == 'marquee':
            y = random.choice([
                random.randint(50, 150),
                random.randint(video_h - 150, video_h - 50)
            ])
            if random.random() < 0.5:
                self.trajectory = get_marquee_trajectory(video_w, y, -self.w, total_frames)
            else:
                self.trajectory = get_marquee_trajectory(-self.w, y, video_w, total_frames)

        elif mode == 'static':
            margin = 30
            x = random.randint(margin, video_w - self.w - margin)
            y = random.randint(margin, video_h - self.h - margin)
            if random.random() < 0.8:
                x = random.choice([margin, video_w - self.w - margin])
                y = random.choice([margin, video_h - self.h - margin])
                x += random.randint(-5, 5)
                y += random.randint(-5, 5)

            self.trajectory = get_bezier_curve((x, y), (x, y), (x, y), (x, y), total_frames)

        else:  # float
            margin = 50
            p0 = (random.randint(margin, video_w - self.w), random.randint(margin, video_h - self.h))
            p3 = (random.randint(margin, video_w - self.w), random.randint(margin, video_h - self.h))
            p1 = (random.randint(0, video_w), random.randint(0, video_h))
            p2 = (random.randint(0, video_w), random.randint(0, video_h))
            self.trajectory = get_bezier_curve(p0, p1, p2, p3, total_frames)

        # === [增强版] 5. 动态呼吸透明度 ===
        # 基础透明度范围稍微降低，让它有机会更透明
        self.alpha_seq = np.ones(total_frames) * random.uniform(0.5, 0.9)

        # 提高呼吸触发概率到 60%
        if random.random() < 0.6:
            # 频率提高：0.1 ~ 0.3 (约 3~10 秒一个周期，在 4 秒视频里会看到明显的明暗变化)
            freq = random.uniform(0.1, 0.3)
            t = np.arange(total_frames)

            # 振幅增强：0.15 (之前的两倍)
            # 这样透明度会在 base +/- 0.15 之间波动，肉眼可见
            self.alpha_seq += 0.15 * np.sin(freq * t)

            # 截断到合法范围，防止过低看不见或过高溢出
            self.alpha_seq = np.clip(self.alpha_seq, 0.1, 1.0)

    def get_render_data(self, frame_idx):
        if frame_idx >= len(self.trajectory): frame_idx = -1
        x, y = self.trajectory[frame_idx]
        return int(x), int(y), self.img_rgb_linear, self.img_alpha_base, self.alpha_seq[frame_idx]


# === 画质退化模拟 ===
def simulate_compression(img_srgb):
    """模拟 JPEG 压缩块效应和模糊"""
    if random.random() < 0.5:
        # JPEG 压缩
        quality = random.randint(40, 90)
        _, encimg = cv2.imencode('.jpg', img_srgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img_srgb = cv2.imdecode(encimg, 1)

    if random.random() < 0.2:
        # 高斯模糊
        k = random.choice([3, 5])
        img_srgb = cv2.GaussianBlur(img_srgb, (k, k), 0)

    return img_srgb


def process_one_video(video_path, factory, subset):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    total_frames_ori = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames_count = CLIP_DURATION * TARGET_FPS

    if total_frames_ori < target_frames_count + 10: return
    start_frame = random.randint(0, total_frames_ori - target_frames_count - 5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # === 构建 1-3 个水印 ===
    num_watermarks = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
    active_watermarks = []
    for _ in range(num_watermarks):
        try:
            wm = WatermarkObject(factory, TARGET_SIZE[0], TARGET_SIZE[1], target_frames_count)
            active_watermarks.append(wm)
        except:
            continue

    if not active_watermarks: return

    # 输出路径
    out_dirs = {
        'input': os.path.join(OUTPUT_ROOT, subset, 'input_video'),
        'gt': os.path.join(OUTPUT_ROOT, subset, 'gt_video'),
        'mask': os.path.join(OUTPUT_ROOT, subset, 'binary_mask'),
    }
    for d in out_dirs.values(): os.makedirs(d, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = {k: cv2.VideoWriter(os.path.join(p, f"{video_name}.mp4"), fourcc, TARGET_FPS, TARGET_SIZE)
               for k, p in out_dirs.items()}

    frames_processed = 0
    while frames_processed < target_frames_count:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, TARGET_SIZE)
        bg_linear = srgb_to_linear(frame)

        final_overlay = np.zeros_like(bg_linear)
        final_alpha = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.float32)

        # 渲染
        for wm in active_watermarks:
            x, y, wm_rgb, wm_a_base, global_a = wm.get_render_data(frames_processed)
            h, w = wm.h, wm.w

            # 边界处理
            y1, y2 = max(0, y), min(TARGET_SIZE[1], y + h)
            x1, x2 = max(0, x), min(TARGET_SIZE[0], x + w)
            wm_y1, wm_y2 = y1 - y, y2 - y
            wm_x1, wm_x2 = x1 - x, x2 - x

            if wm_y2 <= wm_y1 or wm_x2 <= wm_x1: continue

            # 提取图层
            roi_bg = final_overlay[y1:y2, x1:x2]
            roi_alpha_prev = final_alpha[y1:y2, x1:x2]

            wm_part = wm_rgb[wm_y1:wm_y2, wm_x1:wm_x2]
            alpha_part = wm_a_base[wm_y1:wm_y2, wm_x1:wm_x2] * global_a
            alpha_expanded = np.expand_dims(alpha_part, axis=2)

            # 正确混合
            roi_new_overlay = wm_part * alpha_expanded + roi_bg * (1 - alpha_expanded)
            roi_new_alpha = alpha_part + roi_alpha_prev * (1 - alpha_part)

            final_overlay[y1:y2, x1:x2] = roi_new_overlay
            final_alpha[y1:y2, x1:x2] = np.clip(roi_new_alpha, 0, 1)

        # 合成
        final_alpha_expanded = np.expand_dims(final_alpha, axis=2)
        blended_linear = final_overlay * final_alpha_expanded + bg_linear * (1 - final_alpha_expanded)
        blended_srgb = linear_to_srgb(blended_linear).astype(np.uint8)

        # [画质退化]
        blended_srgb = simulate_compression(blended_srgb)

        # 掩码
        binary_mask = (final_alpha > 0.01).astype(np.uint8) * 255
        binary_mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        writers['input'].write(blended_srgb)
        writers['gt'].write(frame)
        writers['mask'].write(binary_mask_rgb)

        frames_processed += 1

    cap.release()
    for w in writers.values(): w.release()


def main():
    all_videos = glob.glob(os.path.join(SOURCE_DIR, "*.mp4"))
    if not all_videos: print("No source videos!"); return

    # 必须传入字体文件夹，否则只能生成英文
    factory = DynamicWatermarkFactory(FONT_DIR)

    random.shuffle(all_videos)
    split = int(len(all_videos) * TRAIN_RATIO)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for v in all_videos[:split]: executor.submit(process_one_video, v, factory, 'train')
        for v in all_videos[split:]: executor.submit(process_one_video, v, factory, 'test')


if __name__ == "__main__":
    main()