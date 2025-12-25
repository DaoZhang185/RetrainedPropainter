import cv2
import os
import glob
from concurrent.futures import ThreadPoolExecutor

# 数据集根目录
DATASET_ROOT = "../datasets/result_videos/academic_dataset_v1"


def process_video(mp4_path):
    # 构建输出文件夹路径
    # 例如: train/input_video/vid1.mp4 -> train_frames/input_video/vid1/0000.jpg
    parts = mp4_path.split(os.sep)
    # 替换 datasets/result_videos/academic_dataset_v1 -> datasets/result_videos/academic_dataset_v1_frames
    new_root = DATASET_ROOT + "_frames"

    subset = parts[-3]  # train or test
    type_dir = parts[-2]  # input_video, gt_video...
    vid_name = os.path.splitext(parts[-1])[0]

    out_dir = os.path.join(new_root, subset, type_dir, vid_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(mp4_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        # 保存为 png 或 jpg
        cv2.imwrite(os.path.join(out_dir, f"{count:05d}.jpg"), frame)
        count += 1
    cap.release()
    print(f"Unpacked: {vid_name} ({type_dir})")


if __name__ == "__main__":
    # 找到所有 mp4
    all_mp4 = glob.glob(os.path.join(DATASET_ROOT, "*", "*", "*.mp4"))
    print(f"Found {len(all_mp4)} videos. Unpacking...")

    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(process_video, all_mp4)