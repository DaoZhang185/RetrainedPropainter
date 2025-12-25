import os
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ================= 配置区域 =================
# 字体目录 (必须包含你刚下载的那个90MB的大字体文件)
FONT_DIR = "../datasets/fonts"
# 输出目录
OUTPUT_DIR = "test_results"

# 强制测试的 6 组样本
TEST_CASES = {
    "01_English": "CONFIDENTIAL 2024",
    "02_Chinese": "绝密：内部影像资料",
    "03_Japanese": "極秘：無断転載禁止",  # 包含汉字和平假名/片假名
    "04_Korean": "견본：복사 금지",  # 韩语谚文
    "05_Russian": "Секретно: Образец",  # 西里尔字母
    "06_Special": "© 2025 Studio | ID: #9527"
}


# ===========================================

def generate_test_image(text, font_path, save_name):
    # 1. 加载字体 (字号设大一点方便看)
    font_size = 100
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"❌ 字体加载失败: {e}")
        return

    # 2. 计算文字大小
    dummy_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 3. 创建画布 (留足边距)
    margin = 50
    canvas_w = text_w + margin * 2
    canvas_h = text_h + margin * 2

    # 背景设为半透明黑色，文字设为白色，模拟水印效果
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 200))
    draw = ImageDraw.Draw(img)

    # 4. 居中绘制 (应用坐标修复逻辑)
    x_offset = -bbox[0]
    y_offset = -bbox[1]
    draw.text((margin + x_offset, margin + y_offset), text, font=font, fill=(255, 255, 255, 255))

    # 5. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, save_name)
    img.save(out_path)
    print(f"✅ 生成成功: {out_path} | 内容: {text}")


def main():
    # 1. 寻找字体
    fonts = glob.glob(os.path.join(FONT_DIR, "*.[ot]tf"))
    if not fonts:
        print(f"❌ 错误：在 {FONT_DIR} 下没找到任何字体文件！")
        print("   请确保你已经把下载的 SourceHanSansK-Bold.otf (或其他Pan-CJK字体) 放进去了。")
        return

    # 默认取第一个找到的字体
    # 如果你有多个，想指定某一个，可以直接把 font_path 写死
    font_path = fonts[0]
    print(f"🔤 正在使用字体: {os.path.basename(font_path)}\n")

    # 2. 循环生成 6 张图
    for key, text in TEST_CASES.items():
        filename = f"{key}.png"
        generate_test_image(text, font_path, filename)

    print(f"\n🎉 全部完成！请去 {OUTPUT_DIR} 文件夹查看图片。")
    print("   如果韩语或日语显示为方框(□)，说明你的字体包不是全能版(Pan-CJK)。")


if __name__ == "__main__":
    main()