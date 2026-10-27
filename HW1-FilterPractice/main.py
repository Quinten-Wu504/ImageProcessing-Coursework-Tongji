import cv2
import numpy as np
import matplotlib.pyplot as plt
from filter_utils import ideal_filter, gaussian_filter, butterworth_filter


def add_gaussian_noise(img, mean=0, sigma=25):
    """为图像添加高斯噪声"""
    # 将图像转换为浮点型（0-1范围）
    img_normalized = img.astype(np.float32) / 255.0

    # 生成高斯噪声（与图像同尺寸）
    h, w = img.shape
    noise = np.random.normal(mean, sigma / 255.0, (h, w))

    # 添加噪声并裁剪到合理范围
    noisy_img = img_normalized + noise
    noisy_img = np.clip(noisy_img, 0, 1)

    # 转换回uint8类型（0-255范围）
    return (noisy_img * 255).astype(np.uint8)


def test_filter(filter_func, test_img_path, filter_type, noise_sigma=25):
    print(f"{filter_type} Result: ")
    # 读取原始图像
    original_img = cv2.imread(test_img_path, 0)  # 以灰度模式读取

    # 添加高斯噪声
    noisy_img = add_gaussian_noise(original_img, sigma=noise_sigma)

    # 低频滤波器（能量90%和60%）
    low_pass_90, mask_low_90 = filter_func(noisy_img, 90, 'low')
    low_pass_60, mask_low_60 = filter_func(noisy_img, 60, 'low')

    # 高频滤波器（能量90%和60%）
    high_pass_90, mask_high_90 = filter_func(noisy_img, 90, 'high')
    high_pass_60, mask_high_60 = filter_func(noisy_img, 60, 'high')

    # 显示结果
    fig = plt.figure(figsize=(15, 10))
    fig.canvas.manager.set_window_title(f"{filter_type} Result")

    # 原图
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.imread(test_img_path, 0), cmap='gray')
    plt.title(f'Original Image')

    # 噪声图
    plt.subplot(2, 3, 4)
    plt.imshow(noisy_img, cmap='gray')
    plt.title(f'Noisy Image')

    # 低频90%
    plt.subplot(2, 3, 2)
    plt.imshow(low_pass_90, cmap='gray')
    plt.title(f'Low Pass (90% Energy)')

    # 低频60%
    plt.subplot(2, 3, 3)
    plt.imshow(low_pass_60, cmap='gray')
    plt.title(f'Low Pass (60% Energy)')

    # 高频90%
    plt.subplot(2, 3, 5)
    plt.imshow(high_pass_90, cmap='gray')
    plt.title(f'High Pass (90% Energy)')

    # 高频60%
    plt.subplot(2, 3, 6)
    plt.imshow(high_pass_60, cmap='gray')
    plt.title(f'High Pass (60% Energy)')

    plt.tight_layout()
    plt.show()

    print()


if __name__ == '__main__':
    # 示例使用（替换为实际图像路径）
    img_path = './TestData/test_high_freq.jpg'  # 请确保图像路径正确

    test_filter(ideal_filter, img_path, filter_type='Ideal Filter')
    test_filter(gaussian_filter, img_path, filter_type='Gaussian Filter')
    test_filter(butterworth_filter, img_path, filter_type='Butterworth Filter')
