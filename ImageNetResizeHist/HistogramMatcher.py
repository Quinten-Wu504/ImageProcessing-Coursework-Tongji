import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


class HistogramMatcher:
    def __init__(self, reference_image, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.reference_image = reference_image
        self.ref_histograms = self._compute_reference_histograms()

    def _compute_histogram_and_cdf(self, img):
        B, C, H, W = img.shape
        img = img.to(self.device)

        # 确保图像是 uint8 格式，范围 [0, 255]
        if torch.is_floating_point(img):
            if img.max() <= 1.0:
                img = (img * 255).round()
            img = img.clamp(0, 255).to(torch.uint8)
        else:
            img = img.clamp(0, 255).to(torch.uint8)

        img_flat = img.view(B, C, -1)  # [B, C, H * W]
        hist_nums = 256
        hist = torch.zeros(B, C, hist_nums, device=self.device)

        for i in range(hist_nums):
            hist[:, :, i] = (img_flat == i).sum(dim=-1)

        cdf = hist.cumsum(dim=-1)  # [B, C, 256]
        cdf = cdf / (cdf[:, :, -1:] + 1e-10)  # 归一化到 [0, 1]
        return hist, cdf

    def _compute_reference_histograms(self):
        ref_np = np.array(self.reference_image)[None, ...]  # [1, H, W, 3]
        ref_histograms = {}

        # RGB
        ref_rgb = torch.from_numpy(ref_np).permute(0, 3, 1, 2).to(self.device)  # [1, 3, H, W]
        ref_histograms['rgb'] = self._compute_histogram_and_cdf(ref_rgb)

        # YUV
        ref_yuv = np.stack([cv2.cvtColor(ref_np[i], cv2.COLOR_RGB2YUV) for i in range(1)], axis=0)
        ref_yuv = torch.from_numpy(ref_yuv).permute(0, 3, 1, 2).to(self.device)  # [1, 3, H, W]
        ref_histograms['yuv'] = self._compute_histogram_and_cdf(ref_yuv[:, 0:1, :, :])  # 只取 Y 通道

        # HSV
        ref_hsv = np.stack([cv2.cvtColor(ref_np[i], cv2.COLOR_RGB2HSV) for i in range(1)], axis=0)
        ref_hsv = torch.from_numpy(ref_hsv).permute(0, 3, 1, 2).to(self.device)  # [1, 3, H, W]
        ref_histograms['hsv'] = self._compute_histogram_and_cdf(ref_hsv[:, 2:3, :, :])  # 只取 V 通道

        # GRAY
        ref_gray = np.stack([cv2.cvtColor(ref_np[i], cv2.COLOR_RGB2GRAY)[:, :, None] for i in range(1)], axis=0)
        ref_gray = torch.from_numpy(ref_gray).permute(0, 3, 1, 2).to(self.device)  # [1, 1, H, W]
        ref_histograms['gray'] = self._compute_histogram_and_cdf(ref_gray)

        return ref_histograms

    def _histogram_match(self, img, ref_hist, ref_cdf):
        _, src_cdf = self._compute_histogram_and_cdf(img)
        B, C, H, W = img.shape
        img_flat = img.view(B, C, -1)  # [B, C, H * W]

        # 完全向量化构建映射表
        src_cdf_expanded = src_cdf[:, :, :, None]  # [B, C, 256, 1]
        ref_cdf_expanded = ref_cdf[:, :, None, :]  # [1, C, 1, 256]
        diff = torch.abs(src_cdf_expanded - ref_cdf_expanded)  # [B, C, 256, 256]
        mapping = torch.argmin(diff, dim=-1).to(torch.uint8)  # [B, C, 256]

        # 应用映射并确保输出为 uint8
        img_matched = mapping.gather(2, img_flat.to(torch.long))  # [B, C, H * W]
        img_matched = img_matched.view(B, C, H, W).to(torch.uint8)  # 确保输出为 uint8
        return img_matched

    def _match_rgb(self, img_rgb):
        img_rgb = torch.from_numpy(img_rgb).permute(0, 3, 1, 2).to(self.device)  # [B, 3, H, W]
        ref_hist, ref_cdf = self.ref_histograms['rgb']
        img_matched = self._histogram_match(img_rgb, ref_hist, ref_cdf)
        return img_matched.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)

    def _match_yuv(self, img_rgb):
        B, H, W, _ = img_rgb.shape
        img_yuv = np.stack([cv2.cvtColor(img_rgb[i], cv2.COLOR_RGB2YUV) for i in range(B)], axis=0)
        img_yuv = torch.from_numpy(img_yuv).permute(0, 3, 1, 2).to(self.device)  # [B, 3, H, W]
        y_channel = img_yuv[:, 0:1, :, :]  # 只取 Y 通道
        ref_hist, ref_cdf = self.ref_histograms['yuv']
        y_matched = self._histogram_match(y_channel, ref_hist, ref_cdf)
        img_yuv_matched = img_yuv.clone()
        img_yuv_matched[:, 0:1, :, :] = y_matched
        img_yuv_matched_np = img_yuv_matched.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        img_rgb_matched = np.stack([cv2.cvtColor(img_yuv_matched_np[i], cv2.COLOR_YUV2RGB) for i in range(B)])
        return img_rgb_matched

    def _match_hsv(self, img_rgb):
        B, H, W, _ = img_rgb.shape
        img_hsv = np.stack([cv2.cvtColor(img_rgb[i], cv2.COLOR_RGB2HSV) for i in range(B)], axis=0)
        img_hsv = torch.from_numpy(img_hsv).permute(0, 3, 1, 2).to(self.device)  # [B, 3, H, W]
        v_channel = img_hsv[:, 2:3, :, :]  # 只取 V 通道
        ref_hist, ref_cdf = self.ref_histograms['hsv']
        v_matched = self._histogram_match(v_channel, ref_hist, ref_cdf)
        img_hsv_matched = img_hsv.clone()
        img_hsv_matched[:, 2:3, :, :] = v_matched
        img_hsv_matched_np = img_hsv_matched.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        img_rgb_matched = np.stack([cv2.cvtColor(img_hsv_matched_np[i], cv2.COLOR_HSV2RGB) for i in range(B)])
        return img_rgb_matched

    def _match_gray(self, img_rgb):
        B, H, W, _ = img_rgb.shape
        img_gray = np.stack([cv2.cvtColor(img_rgb[i], cv2.COLOR_RGB2GRAY)[:, :, None] for i in range(B)], axis=0)
        img_gray = torch.from_numpy(img_gray).permute(0, 3, 1, 2).to(self.device)  # [B, 1, H, W]
        ref_hist, ref_cdf = self.ref_histograms['gray']
        img_gray_matched = self._histogram_match(img_gray, ref_hist, ref_cdf)
        return img_gray_matched.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)

    def match(self, img_rgb, mode='rgb'):
        if mode == 'rgb':
            return self._match_rgb(img_rgb)
        elif mode == 'yuv':
            return self._match_yuv(img_rgb)
        elif mode == 'hsv':
            return self._match_hsv(img_rgb)
        elif mode == 'gray':
            return self._match_gray(img_rgb)
        else:
            raise ValueError(f'Unsupported mode: {mode}')


def histogram_match_all_images(reference_image_path, batch_size=16, input_dir='./dataset/ImageNet2000_Resized', output_dir='./dataset/ImageNet2000_Matched'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference_image = np.array(Image.open(reference_image_path).convert("RGB"))
    matcher = HistogramMatcher(reference_image=reference_image, device=device)
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

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Match Processing Batches"):
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
        matched_results = {mode: matcher.match(batch_np, mode=mode) for mode in modes}

        for b_idx, name in enumerate(valid_filenames):
            save_dir = os.path.join(output_dir, name)
            os.makedirs(save_dir, exist_ok=True)

            # 保存原图和参考图
            original_rgb = original_images[b_idx]
            original_path = os.path.join(save_dir, 'original.JPEG')
            cv2.imwrite(original_path, cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))
            ref_path = os.path.join(save_dir, 'reference.JPEG')
            cv2.imwrite(ref_path, cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR))

            # 保存匹配图
            combined_images = []

            def add_label(img, label):
                img_copy = img.copy()
                cv2.putText(img_copy, label, (10, 30), font, font_scale, font_color, thickness, line_type)
                return img_copy

            # 原图和参考图添加标签
            labeled_original = add_label(original_rgb, 'original')
            labeled_ref = add_label(reference_image, 'reference')
            combined_images.extend([labeled_original, labeled_ref])

            for mode in modes:
                img = matched_results[mode][b_idx]
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

            # 拼接六张图像（从左到右），保存 combined.JPEG
            try:
                combined = np.hstack(combined_images)  # 横向拼接
                combined_path = os.path.join(save_dir, 'combined.JPEG')
                cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Failed to combine images for {name}: {e}")


if __name__ == "__main__":
    src_folder = "./dataset/ImageNet2000_Resized"
    dst_folder = "./dataset/ImageNet2000_Matched"
    reference_image_path = "./dataset/reference.JPEG"  # 替换为你的参考图像路径
    batch = 64

    histogram_match_all_images(
        reference_image_path=reference_image_path,
        batch_size=batch,
        input_dir=src_folder,
        output_dir=dst_folder,
    )
    