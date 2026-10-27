import os
import cv2
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from skimage.metrics import structural_similarity as compare_ssim

from GenerateImageNet2000 import sample_imagenet_val
from ImageResizer import resize_all_images
from HistogramEqualizer import histogram_equalize_all_images
from HistogramSpecifier import histogram_specify_all_images
from HistogramMatcher import histogram_match_all_images
from CLAHEProcessor import clahe_process_all_images


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_mse_psnr(img1, img2):
    try:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 0, float('inf')
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return mse, psnr
    except Exception as e:
        logging.error(f"错误：MSE/PSNR 计算失败: {e}")
        return np.nan, np.nan


def calculate_contrast(img):
    try:
        return np.std(img)
    except Exception as e:
        logging.error(f"错误：对比度计算失败: {e}")
        return np.nan


def calculate_image_entropy(img):
    try:
        hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256], density=True)
        return entropy(hist, base=2)
    except Exception as e:
        logging.error(f"错误：熵计算失败: {e}")
        return np.nan


def evaluate_images(input_dir, methods=['CLAHE', 'Equalized', 'Specified', 'Matched']):
    # 初始化结果字典：{method: {img_type: {metric: []}}}
    results = {
        method: {
            img_type: {'mse': [], 'psnr': [], 'ssim': [], 'contrast': [], 'entropy': []}
            for img_type in ['gray', 'rgb', 'yuv', 'hsv']
        } for method in methods
    }
    detailed_results = []  # 用于详细 CSV 输出
    img_types = ['gray', 'rgb', 'yuv', 'hsv']
    file_extensions = ['JPEG', 'jpeg']

    # 遍历，显示进度条
    for img_id in tqdm(range(1, 2001), desc="处理图像", unit="image"):
        img_name = f"{img_id:04d}"
        # 尝试加载原始图像
        original_path = None
        for ext in file_extensions:
            temp_path = os.path.join(input_dir, 'ImageNet2000_Resized', f'{img_name}.{ext}')
            if os.path.exists(temp_path):
                original_path = temp_path
                break

        if not original_path:
            logging.warning(f"未找到原始图像: {img_name}")
            continue

        original_img = cv2.imread(original_path, cv2.IMREAD_COLOR)
        if original_img is None:
            logging.error(f"无法加载原始图像: {original_path}")
            continue

        original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        for method in methods:
            folder_path = os.path.join(input_dir, f'ImageNet2000_{method}', img_name)
            if not os.path.exists(folder_path):
                logging.warning(f"未找到文件夹: {folder_path}")
                continue

            for img_type in img_types:
                img_path = None
                for ext in file_extensions:
                    temp_path = os.path.join(folder_path, f'{img_type}.{ext}')
                    if os.path.exists(temp_path):
                        img_path = temp_path
                        break

                if not img_path:
                    logging.warning(f"未找到增强图像: {method}/{img_name}/{img_type}")
                    continue

                enhanced_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if enhanced_img is None:
                    logging.error(f"无法加载增强图像: {img_path}")
                    continue

                if enhanced_img.shape != original_img.shape:
                    logging.warning(f"图像尺寸不匹配: {img_path} ({enhanced_img.shape} vs {original_img.shape})")
                    continue

                enhanced_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

                mse, psnr = calculate_mse_psnr(original_img, enhanced_img)
                ssim = compare_ssim(original_gray, enhanced_gray, multichannel=False, data_range=255)
                contrast = calculate_contrast(enhanced_gray)
                entropy_val = calculate_image_entropy(enhanced_gray)

                results[method][img_type]['mse'].append(mse)
                results[method][img_type]['psnr'].append(psnr)
                results[method][img_type]['ssim'].append(ssim)
                results[method][img_type]['contrast'].append(contrast)
                results[method][img_type]['entropy'].append(entropy_val)

                detailed_results.append({
                    'ImageID': img_name,
                    'Method': method,
                    'ImageType': img_type,
                    'MSE': mse,
                    'PSNR': psnr,
                    'SSIM': ssim,
                    'Contrast': contrast,
                    'Entropy': entropy_val
                })

    # 生成并打印汇总表格
    metrics = ['mse', 'psnr', 'ssim', 'contrast', 'entropy']
    metric_names = {
        'mse': 'MSE',
        'psnr': 'PSNR',
        'ssim': 'SSIM',
        'contrast': '对比度',
        'entropy': '熵'
    }
    summary_tables = {}

    for metric in metrics:
        # 创建表格：行是方法，列是颜色空间
        table_data = {}
        for method in methods:
            table_data[method] = {}
            for img_type in img_types:
                values = [v for v in results[method][img_type][metric] if not np.isnan(v)]
                avg = np.mean(values) if values else np.nan
                table_data[method][img_type] = avg
        summary_tables[metric] = pd.DataFrame(table_data).T  # 转置使方法为行，颜色空间为列

        print(f"\n{metric_names[metric]} 表格：")
        print(summary_tables[metric].to_string(float_format="%.4f"))
        print(
            f"（基于有效图像数：{[len([v for v in results[method][img_type][metric] if not np.isnan(v)]) for method in methods for img_type in img_types]}）")

    # 保存详细结果到 CSV
    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv(os.path.join(input_dir, 'evaluation_results.csv'), index=False)
    logging.info(f"详细结果已保存至 {os.path.join(input_dir, 'evaluation_results.csv')}")

    # 保存汇总表格到 CSV
    with pd.ExcelWriter(os.path.join(input_dir, 'evaluation_summary.xlsx')) as writer:
        for metric in metrics:
            summary_tables[metric].to_excel(writer, sheet_name=metric_names[metric], float_format="%.4f")
    logging.info(f"汇总表格已保存至 {os.path.join(input_dir, 'evaluation_summary.xlsx')}")

    return results


if __name__ == "__main__":
    # Step 1: 从ImageNet验证集中采样2000张图像
    sample_imagenet_val(
        source_dir="./dataset/ImageNetVal",
        target_dir="./dataset/ImageNet2000",
        sample_count=2000
    )
    print("Step 1 完成：采样完成。")

    # Step 2: Resize图像到256x256
    resize_all_images(
        source_dir="./dataset/ImageNet2000",
        target_dir="./dataset/ImageNet2000_Resized",
        height=256,
        width=256,
        mode='bilinear'
    )
    print("Step 2 完成：Resize完成。")

    # Step 3: 直方图均衡化
    histogram_equalize_all_images(
        batch_size=64,
        input_dir="./dataset/ImageNet2000_Resized",
        output_dir="./dataset/ImageNet2000_Equalized",
    )
    print("Step 3 完成：直方图均衡化完成。")

    # Step 4: 直方图规定化
    histogram_specify_all_images(
        batch_size=64,
        alpha=0.05,
        input_dir="./dataset/ImageNet2000_Resized",
        output_dir="./dataset/ImageNet2000_Specified",
    )
    print("Step 4 完成：直方图指定完成。")

    # Step 5: 直方图匹配（以指定参考图像为基准）
    histogram_match_all_images(
        reference_image_path="./dataset/reference.JPEG",
        batch_size=64,
        input_dir="./dataset/ImageNet2000_Resized",
        output_dir="./dataset/ImageNet2000_Matched",
    )
    print("Step 5 完成：直方图匹配完成。")

    # Step 6: CLAHE增强
    clahe_process_all_images(
        batch_size=64,
        input_dir="./dataset/ImageNet2000_Resized",
        output_dir="./dataset/ImageNet2000_CLAHE",
    )
    print("Step 6 完成：CLAHE完成。")

    # Step 7: 评估增强结果
    evaluate_images(input_dir="./dataset")
    print("Step 7 完成：评估完成。")
