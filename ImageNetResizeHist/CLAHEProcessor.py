import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


class CLAHEProcessor:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        初始化CLAHE处理器。

        Args:
            clip_limit: 对比度限制阈值，控制增强幅度，默认2.0
            tile_grid_size: 瓦片大小，元组形式 (rows, cols)，默认(8, 8)
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def _clahe_rgb(self, img_rgb):
        """
        对RGB图像的每个通道应用CLAHE。

        Args:
            img_rgb: 输入图像，numpy数组，形状为[B, H, W, 3]

        Returns:
            处理后的图像，numpy数组，形状为[B, H, W, 3]
        """
        B, H, W, _ = img_rgb.shape
        img_rgb_out = np.zeros_like(img_rgb, dtype=np.uint8)

        for b in range(B):
            # 分离RGB通道
            channel_r, channel_g, channel_b = cv2.split(img_rgb[b])
            # 对每个通道应用CLAHE
            r_clahe = self.clahe.apply(channel_r)
            g_clahe = self.clahe.apply(channel_g)
            b_clahe = self.clahe.apply(channel_b)
            # 合并通道
            img_rgb_out[b] = cv2.merge((r_clahe, g_clahe, b_clahe))

        return img_rgb_out

    def _clahe_yuv(self, img_rgb):
        """
        对YUV图像的Y通道应用CLAHE。

        Args:
            img_rgb: 输入图像，numpy数组，形状为[B, H, W, 3]

        Returns:
            处理后的图像，numpy数组，形状为[B, H, W, 3]
        """
        B, H, W, _ = img_rgb.shape
        img_rgb_out = np.zeros_like(img_rgb, dtype=np.uint8)

        for b in range(B):
            # 转换为YUV
            img_yuv = cv2.cvtColor(img_rgb[b], cv2.COLOR_RGB2YUV)
            # 分离YUV通道
            y, u, v = cv2.split(img_yuv)
            # 对Y通道应用CLAHE
            y_clahe = self.clahe.apply(y)
            # 合并通道并转回RGB
            img_yuv_clahe = cv2.merge((y_clahe, u, v))
            img_rgb_out[b] = cv2.cvtColor(img_yuv_clahe, cv2.COLOR_YUV2RGB)

        return img_rgb_out

    def _clahe_hsv(self, img_rgb):
        """
        对HSV图像的V通道应用CLAHE。

        Args:
            img_rgb: 输入图像，numpy数组，形状为[B, H, W, 3]

        Returns:
            处理后的图像，numpy数组，形状为[B, H, W, 3]
        """
        B, H, W, _ = img_rgb.shape
        img_rgb_out = np.zeros_like(img_rgb, dtype=np.uint8)

        for b in range(B):
            # 转换为HSV
            img_hsv = cv2.cvtColor(img_rgb[b], cv2.COLOR_RGB2HSV)
            # 分离HSV通道
            h, s, v = cv2.split(img_hsv)
            # 对V通道应用CLAHE
            v_clahe = self.clahe.apply(v)
            # 合并通道并转回RGB
            img_hsv_clahe = cv2.merge((h, s, v_clahe))
            img_rgb_out[b] = cv2.cvtColor(img_hsv_clahe, cv2.COLOR_HSV2RGB)

        return img_rgb_out

    def _clahe_gray(self, img_rgb):
        """
        对灰度图像应用CLAHE。

        Args:
            img_rgb: 输入图像，numpy数组，形状为[B, H, W, 3]

        Returns:
            处理后的灰度图像，numpy数组，形状为[B, H, W, 1]
        """
        B, H, W, _ = img_rgb.shape
        img_gray_out = np.zeros((B, H, W, 1), dtype=np.uint8)

        for b in range(B):
            # 转换为灰度
            img_gray = cv2.cvtColor(img_rgb[b], cv2.COLOR_RGB2GRAY)
            # 应用CLAHE
            img_gray_out[b, :, :, 0] = self.clahe.apply(img_gray)

        return img_gray_out

    def process(self, img_rgb, mode='rgb'):
        """
        执行CLAHE处理。

        Args:
            img_rgb: 输入图像，numpy数组，形状为[B, H, W, 3]
            mode: 颜色空间，'rgb', 'yuv', 'hsv', 'gray'之一

        Returns:
            处理后的图像，numpy数组
        """
        # 确保输入为uint8
        img_rgb = img_rgb.astype(np.uint8)

        if mode == 'rgb':
            return self._clahe_rgb(img_rgb)
        elif mode == 'yuv':
            return self._clahe_yuv(img_rgb)
        elif mode == 'hsv':
            return self._clahe_hsv(img_rgb)
        elif mode == 'gray':
            return self._clahe_gray(img_rgb)
        else:
            raise ValueError(f'不支持的模式: {mode}')


def clahe_process_all_images(batch_size=16, input_dir='./dataset/ImageNet2000_Resized',
                             output_dir='./dataset/ImageNet2000_CLAHE'):
    """
    对所有图像执行CLAHE处理，并保存结果。

    Args:
        batch_size: 批量大小
        input_dir: 输入图像目录
        output_dir: 输出图像目录
    """
    processor = CLAHEProcessor(clip_limit=2.0, tile_grid_size=(8, 8))
    modes = ['rgb', 'yuv', 'hsv', 'gray']

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 0, 0)  # 蓝色（BGR）
    thickness = 2
    line_type = cv2.LINE_AA

    os.makedirs(output_dir, exist_ok=True)

    image_paths = [
        os.path.join(input_dir, fname)
        for fname in sorted(os.listdir(input_dir))
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLAHE Processing Batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        valid_filenames = []
        original_images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                img_np = np.array(img)
                batch_images.append(img_np)
                original_images.append(img_np)
                fname = os.path.splitext(os.path.basename(path))[0]
                valid_filenames.append(fname)
            except Exception as e:
                print(f"无法读取 {path}: {e}")

        if not batch_images:
            continue

        batch_np = np.stack(batch_images, axis=0)  # [B, H, W, 3]

        # 保存所有模式的图像
        clahe_results = {mode: processor.process(batch_np, mode=mode) for mode in modes}

        for b_idx, name in enumerate(valid_filenames):
            save_dir = os.path.join(output_dir, name)
            os.makedirs(save_dir, exist_ok=True)

            # 保存原图
            original_rgb = original_images[b_idx]
            original_path = os.path.join(save_dir, 'original.JPEG')
            cv2.imwrite(original_path, cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))

            # 保存CLAHE处理结果
            combined_images = []

            def add_label(img, label):
                img_copy = img.copy()
                cv2.putText(img_copy, label, (10, 30), font, font_scale, font_color, thickness, line_type)
                return img_copy

            # 原图添加标签
            labeled_original = add_label(original_rgb, 'original')
            combined_images.append(labeled_original)

            for mode in modes:
                img = clahe_results[mode][b_idx]
                save_path = os.path.join(save_dir, f'{mode}.JPEG')

                if mode == 'gray':
                    img = img.squeeze(-1)  # [H, W]
                    cv2.imwrite(save_path, img)
                    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    img_color = img

                labeled_img = add_label(img_color, mode)
                combined_images.append(labeled_img)

            # 拼接五张图像（从左到右），保存combined.JPEG
            try:
                combined = np.hstack(combined_images)  # 横向拼接
                combined_path = os.path.join(save_dir, 'combined.JPEG')
                cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"无法拼接图像 {name}: {e}")


if __name__ == "__main__":
    src_folder = "./dataset/ImageNet2000_Resized"
    dst_folder = "./dataset/ImageNet2000_CLAHE"
    batch = 64

    clahe_process_all_images(
        batch_size=batch,
        input_dir=src_folder,
        output_dir=dst_folder,
    )