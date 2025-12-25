import os
import random
import requests
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# ================= 配置区域 =================
# 填入你的 Pexels API Key
PEXELS_API_KEY = "hpXPu6by1Y9smZdxh6k5BXsvI2JGfv9Djslx9jYRdJfKwglRA3wdf5AO"

# 输出目录
OUTPUT_DIR = "../datasets/source_videos"

# 目标数量
TARGET_COUNT = 10000

# 搜索关键词列表 (为了保证多样性，轮询使用这些词)
QUERIES = [
    # === 1. 自然风光 (复杂的纹理与随机运动) ===
    "nature", "forest", "river", "ocean", "mountain", "sky", "clouds", "rain",
    "snow", "desert", "beach", "jungle", "waterfall", "sunset", "sunrise",
    "flowers", "grass", "leaves", "trees", "landscape", "underwater", "waves",
    "winter", "autumn", "summer", "spring", "fog", "mist", "canyon", "cave",
    "cliff", "island", "lake", "park", "garden", "field", "meadow", "stars",

    # === 2. 城市与建筑 (规则线条与刚性物体) ===
    "city", "traffic", "street", "building", "architecture", "road", "highway",
    "bridge", "night city", "skyline", "urban", "skyscraper", "train", "subway",
    "airport", "construction", "industrial", "factory", "neon lights", "lights",
    "cars", "bus", "crowd", "walking", "office", "meeting", "work", "school",
    "library", "stairs", "window", "door", "roof", "tunnel", "wall",

    # === 3. 抽象与背景 (极佳的去水印训练素材 - 纯净或极繁) ===
    "abstract", "texture", "background", "pattern", "blur", "bokeh", "gradient",
    "smoke", "ink", "particles", "light leak", "glitch", "macro", "close up",
    "wood", "stone", "fabric", "paper", "liquid", "oil", "bubbles", "glass",
    "metal", "plastic", "sand", "dirt", "concrete", "marble", "surface",

    # === 4. 科技与商业 (屏幕、文字、人造光) ===
    "technology", "computer", "typing", "coding", "phone", "screen", "keyboard",
    "gaming", "robot", "drone", "science", "lab", "money", "business", "shopping",
    "coffee", "food", "cooking", "restaurant", "party", "concert", "dance",
    "writing", "reading", "working", "presentation", "meeting room",

    # === 5. 动物与生物 (非规则形变) ===
    "animals", "dog", "cat", "bird", "fish", "horse", "wildlife", "zoo",
    "insect", "butterfly", "pet", "puppy", "kitten", "aquarium", "lion",
    "tiger", "monkey", "cow", "sheep", "chicken",

    # === 6. 运动与特殊视角 (大幅度运动与模糊) ===
    "timelapse", "hyperlapse", "slow motion", "aerial", "drone view", "top view",
    "walking pov", "driving", "cycling", "running", "sports", "fitness", "gym",
    "yoga", "swimming", "hiking", "travel", "adventure", "surfing", "skating",
    "skiing", "basketball", "football", "soccer",

    # === 7. 人物与情感 (肤色与复杂遮挡) ===
    "people", "portrait", "woman", "man", "girl", "boy", "baby", "family",
    "friends", "couple", "smiling", "laughing", "talking", "sad", "happy",
    "fashion", "model", "makeup", "hair"
]

# FFmpeg 路径 (如果在环境变量中直接写 'ffmpeg')
FFMPEG_BIN = "ffmpeg"
# ===========================================

headers = {
    "Authorization": PEXELS_API_KEY
}


def get_video_duration(file_path):
    """获取下载后的临时视频时长（防止API给的时长不准）"""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout)
    except:
        return 0.0


def process_single_video(video_meta, save_dir):
    """
    下载 -> 随机裁切5秒 -> 缩放720p -> 保存 -> 删除原片
    """
    video_id = video_meta['id']
    duration_api = video_meta['duration']

    # 1. 筛选 1080p 的源链接
    download_link = None
    width, height = 0, 0

    # Pexels 提供不同尺寸，我们优先找 height=1080 的
    for video_file in video_meta['video_files']:
        if video_file['height'] == 1080:
            download_link = video_file['link']
            width = video_file['width']
            height = video_file['height']
            break

    # 如果没有1080p，找最接近高清的 (例如 2k 或 HD)
    if download_link is None:
        # 排序找到最大的
        files_sorted = sorted(video_meta['video_files'], key=lambda x: x['height'], reverse=True)
        if len(files_sorted) > 0:
            download_link = files_sorted[0]['link']
            width = files_sorted[0]['width']
            height = files_sorted[0]['height']

    if not download_link:
        return False, "No valid link found"

    # 最终文件名
    final_path = os.path.join(save_dir, f"{video_id}_720p.mp4")
    if os.path.exists(final_path):
        return True, "Already exists"

    # 临时文件名
    temp_path = os.path.join(save_dir, f"temp_{video_id}.mp4")

    try:
        # Step A: 下载完整视频 (流式下载节省内存)
        with requests.get(download_link, stream=True, timeout=20) as r:
            r.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Step B: 确定裁剪时间点
        # 获取实际时长 (比API更准)
        real_duration = get_video_duration(temp_path)
        if real_duration < 5.0:
            os.remove(temp_path)
            return False, "Video too short"

        start_time = random.uniform(0, real_duration - 5.0)

        # Step C: FFmpeg 处理 (裁剪 + Resize)
        # scale=-2:720 保持比例，高度定为720，宽度自动计算且为2的倍数
        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{start_time:.2f}",  # 开始时间 (放在 -i 前面加速定位)
            "-i", temp_path,
            "-t", "5",  # 持续 5 秒
            "-vf", "scale=-2:720",  # 滤镜：缩放
            "-c:v", "libx264",  # 编码器
            "-crf", "23",  # 质量控制
            "-c:a", "aac",  # 音频编码
            "-loglevel", "error",  # 减少日志
            final_path
        ]

        subprocess.run(cmd, check=True)

        # Step D: 清理临时文件
        os.remove(temp_path)
        return True, "Success"

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, str(e)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    collected_count = len([n for n in os.listdir(OUTPUT_DIR) if n.endswith(".mp4")])
    print(f"当前已有视频: {collected_count} / {TARGET_COUNT}")

    page = 1
    query_index = 0

    # 线程池 (IO密集型任务，网络下载耗时)
    # 不要开太大，防止被 Pexels 封 IP，建议 4-8
    executor = ThreadPoolExecutor(max_workers=5)
    futures = []

    pbar = tqdm(total=TARGET_COUNT, initial=collected_count)

    while collected_count < TARGET_COUNT:
        # 轮询关键词
        query = QUERIES[query_index % len(QUERIES)]

        print(f"\n正在拉取列表: Query='{query}', Page={page}...")

        try:
            url = f"https://api.pexels.com/videos/search?query={query}&per_page=80&page={page}&orientation=landscape"
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()

            if 'videos' not in data or len(data['videos']) == 0:
                print(f"关键词 '{query}' 已无更多结果，切换下一个...")
                query_index += 1
                page = 1  # 重置页码
                continue

            # ===============================================
            # 如果当前关键词已经搜了超过 15 页，强制换下一个词！
            if page > 15:
                print(f"关键词 '{query}' 搜索太深 (Page {page})，主动切换下一个以保证多样性...")
                query_index += 1
                page = 1
                continue
            # ================================================
            # 提交任务到线程池
            for video_meta in data['videos']:
                if collected_count >= TARGET_COUNT:
                    break

                # 检查时长，小于5秒的直接跳过，不下载
                if video_meta['duration'] < 5:
                    continue

                future = executor.submit(process_single_video, video_meta, OUTPUT_DIR)
                futures.append(future)

            # 处理本批次的结果
            for future in as_completed(futures):
                success, msg = future.result()
                if success:
                    if msg != "Already exists":
                        collected_count += 1
                        pbar.update(1)

                if collected_count >= TARGET_COUNT:
                    break

            futures = []  # 清空本轮
            page += 1
            # 每搜完一轮关键词，换下一个词，增加多样性
            if page % 5 == 0:
                query_index += 1
                page = 1

        except Exception as e:
            print(f"API 请求错误: {e}")
            time.sleep(5)  # 等待几秒重试

    pbar.close()
    print("下载任务完成！")


if __name__ == "__main__":
    # 检查 ffmpeg
    try:
        subprocess.run([FFMPEG_BIN, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("错误: 未找到 ffmpeg，请先安装 ffmpeg 并添加到环境变量。")
        exit(1)

    if "你的_PEXELS_API_KEY" in PEXELS_API_KEY:
        print("错误: 请先在代码中填入你的 Pexels API Key。")
    else:
        main()