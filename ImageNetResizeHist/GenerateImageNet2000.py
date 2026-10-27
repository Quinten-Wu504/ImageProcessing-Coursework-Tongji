import os
import random
import shutil
from tqdm import tqdm  # 用于进度条

def sample_imagenet_val(source_dir, target_dir, sample_count=2000):
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    # 严格筛选 .JPEG 文件（区分大小写）
    all_images = [f for f in os.listdir(source_dir) if f.endswith('.JPEG')]

    if len(all_images) < sample_count:
        raise ValueError(f"源目录中只有 {len(all_images)} 张 .JPEG 图，少于需要的 {sample_count} 张")

    # 随机采样
    sampled_images = random.sample(all_images, sample_count)

    # 拷贝并重命名 + 显示进度条
    for idx, fname in enumerate(tqdm(sampled_images, desc="拷贝图像"), start=1):
        src = os.path.join(source_dir, fname)
        dst = os.path.join(target_dir, f"{idx:04d}.JPEG")
        shutil.copy2(src, dst)

    print(f"\n成功采样并保存了 {sample_count} 张图像到 {target_dir}")

if __name__ == "__main__":
    sample_imagenet_val(
        source_dir="./dataset/ImageNetVal",
        target_dir="./dataset/ImageNet2000",
        sample_count=2000
    )
