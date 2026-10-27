import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


class HistogramSpecifier:
    def __init__(self, alpha=0.05, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.alpha = alpha
        self.target_cdf = self._generate_target_cdf(alpha).to(self.device)  # [256]

    @staticmethod
    def _generate_target_cdf(target_cdf_alpha):
        # S型映射函数，形似 sigmoid，高斯CDF 也类似
        x = torch.arange(256).float()
        cdf = 1 / (1 + torch.exp(-target_cdf_alpha * (x - 128)))
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255  # 归一化到 [0, 255]
        return cdf.round().to(torch.uint8)

    def _histogram_specify(self, img):
        B, C, H, W = img.shape
        img = img.to(self.device)

        # 确保图像是 uint8 格式，范围是 [0, 255]
        if torch.is_floating_point(img):
            if img.max() <= 1.0:  # 假设是 [0, 1] 范围的 float 图像
                img = (img * 255).round()
            img = img.clamp(0, 255).to(torch.uint8)
        else:
            img = img.clamp(0, 255).to(torch.uint8)

        img_flat = img.view(B, C, -1)  # [B, C, H * W]

        # 构建直方图：对每个像素值 [0, 255] 计算出现频次
        hist_nums = 256
        hist = torch.zeros(B, C, hist_nums, device=self.device)

        for i in range(hist_nums):
            hist[:, :, i] = (img_flat == i).sum(dim=-1)

        # 计算累计分布函数（CDF）
        cdf = hist.cumsum(dim=-1)  # [B, C, 256]
        cdf_min = cdf[:, :, 0:1]
        cdf_max = cdf[:, :, -1:]
        cdf = ((cdf - cdf_min) / (cdf_max - cdf_min + 1e-10) * 255).round().to(torch.uint8)  # [B, C, 256]

        # cdf: [B, C, 256]
        # target_cdf: [256]

        # 计算差值矩阵: [B, C, 256, 256]
        # 第一个256是当前图像CDF的每个bin；第二个256是target_cdf所有值
        # 目标：对每个cdf[b, c, i]，找target_cdf中哪个值最接近
        diff = torch.abs(cdf.unsqueeze(-1) - self.target_cdf.view(1, 1, 1, 256))  # [B, C, 256, 256]

        # 在最后一个维度上取最小的索引
        mapping = torch.argmin(diff, dim=-1).to(torch.uint8)  # [B, C, 256]

        specified = mapping.gather(2, img_flat.long()).view(B, C, H, W)
        return specified

    def _specify_rgb(self, img_rgb):
        img_rgb = torch.from_numpy(img_rgb).permute(0, 3, 1, 2).to(self.device)
        specified = self._histogram_specify(img_rgb)
        return specified.permute(0, 2, 3, 1).detach().cpu().numpy()

    def _specify_yuv(self, img_rgb):
        B, H, W, _ = img_rgb.shape

        # 转为 YUV 格式
        img_yuv = np.stack([cv2.cvtColor(img_rgb[i], cv2.COLOR_RGB2YUV) for i in range(B)], axis=0)  # [B, H, W, C]
        img_yuv = torch.from_numpy(img_yuv).permute(0, 3, 1, 2).to(self.device)  # [B, 3, H, W]

        y_channel = img_yuv[:, 0:1, :, :]  # 只取 Y 通道
        y_specified = self._histogram_specify(y_channel)

        img_yuv_specified = img_yuv.clone()
        img_yuv_specified[:, 0:1, :, :] = y_specified  # 替换 Y 通道

        # 转回 numpy [B, H, W, 3] 的 RGB 格式
        img_yuv_specified_np = img_yuv_specified.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        img_rgb_specified = np.stack([cv2.cvtColor(img_yuv_specified_np[i], cv2.COLOR_YUV2RGB) for i in range(B)], axis=0)

        return img_rgb_specified

    def _specify_hsv(self, img_rgb):
        B, H, W, _ = img_rgb.shape

        # 转为 HSV 格式
        img_hsv = np.stack([cv2.cvtColor(img_rgb[i], cv2.COLOR_RGB2HSV) for i in range(B)], axis=0)
        img_hsv = torch.from_numpy(img_hsv).permute(0, 3, 1, 2).to(self.device)  # [B, 3, H, W]

        v_channel = img_hsv[:, 2:3, :, :]  # 只取 V 通道
        v_specified = self._histogram_specify(v_channel)

        img_hsv_specified = img_hsv.clone()
        img_hsv_specified[:, 2:3, :, :] = v_specified  # 替换 V 通道

        # 转回 numpy [B, H, W, 3] 的 RGB 格式
        img_hsv_specified_np = img_hsv_specified.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        img_rgb_specified = np.stack([cv2.cvtColor(img_hsv_specified_np[i], cv2.COLOR_HSV2RGB) for i in range(B)], axis=0)
        return img_rgb_specified

    def _specify_gray(self, img_rgb):
        B, H, W, _ = img_rgb.shape

        # 将 RGB 转化为灰度图（OpenCV 会输出 [H, W]，我们再扩展到 [H, W, 1])
        img_gray = np.stack([cv2.cvtColor(img_rgb[i], cv2.COLOR_RGB2GRAY)[:, :, None] for i in range(B)], axis=0)  # [B, H, W, 1]
        img_gray = torch.from_numpy(img_gray).permute(0, 3, 1, 2).to(self.device)  # [B, 1, H, W]

        img_gray_specified = self._histogram_specify(img_gray)
        img_gray_specified = img_gray_specified.permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, H, W, 1]

        return img_gray_specified  # 返回灰度图，形状为 [B, H, W, 1]

    def specify(self, img_rgb, mode='rgb'):
        if mode == 'rgb':
            return self._specify_rgb(img_rgb)
        elif mode == 'yuv':
            return self._specify_yuv(img_rgb)
        elif mode == 'hsv':
            return self._specify_hsv(img_rgb)
        elif mode == 'gray':
            return self._specify_gray(img_rgb)
        else:
            raise ValueError(f'Unsupported mode: {mode}')


def histogram_specify_all_images(batch_size=16, alpha=0.05, input_dir='./dataset/ImageNet2000_Resized', output_dir='./dataset/ImageNet2000_Specified'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    specifier = HistogramSpecifier(alpha=alpha, device=device)
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

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Specify Processing Batches"):
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
                print(f"Failed to read {path}: {e}")

        if not batch_images:
            continue

        batch_np = np.stack(batch_images, axis=0)  # [B, H, W, 3]

        # 保存所有模式的图像
        specified_results = {mode: specifier.specify(batch_np, mode=mode) for mode in modes}

        for b_idx, name in enumerate(valid_filenames):
            save_dir = os.path.join(output_dir, name)
            os.makedirs(save_dir, exist_ok=True)

            # 保存原图
            original_rgb = original_images[b_idx]
            original_path = os.path.join(save_dir, 'original.JPEG')
            cv2.imwrite(original_path, cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))

            # 保存均衡化图
            combined_images = []  # 用于拼接

            def add_label(img, label):
                img_copy = img.copy()
                cv2.putText(img_copy, label, (10, 30), font, font_scale, font_color, thickness, line_type)
                return img_copy

            # 原图添加标签
            labeled_original = add_label(original_rgb, 'original')
            combined_images.append(labeled_original)

            for mode in modes:
                img = specified_results[mode][b_idx]
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

            # 拼接五张图像 (从左到右)，保存 combined.JPEG
            try:
                combined = np.hstack(combined_images)  # 横向拼接
                combined_path = os.path.join(save_dir, 'combined.JPEG')
                cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Failed to combine images for {name}: {e}")


if __name__ == "__main__":
    src_folder = "./dataset/ImageNet2000_Resized"
    dst_folder = "./dataset/ImageNet2000_Specified"

    batch = 64
    alpha = 0.05

    histogram_specify_all_images(
        batch_size=batch,
        alpha=alpha,
        input_dir=src_folder,
        output_dir=dst_folder,
    )
