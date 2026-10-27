import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(channel):
    """手写直方图均衡化函数（单通道）"""
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-10)
    return cdf_normalized.astype(np.uint8)[channel]


def result_visualization(image_rgb, result_image, channel, channel_equalized, mode='YUV'):
    if mode is 'YUV':
        channel_name = 'Y Channel'
    elif mode is 'HSV':
        channel_name = 'V Channel'
    elif mode is 'GRAY':
        channel_name = 'Gray'
    else:
        raise ValueError("Invalid mode.")

    plt.figure(figsize=(15, 10))

    # 原图与处理之后的图的对比
    img_comparison(image_rgb=image_rgb, result_image=result_image, mode=mode)
    # 原图与处理之后的图的直方图对比
    histogram_comparison(image_rgb=image_rgb, result_image=result_image)
    # 处理的通道的前后对比
    equalized_channel_comparison(channel=channel, channel_equalized=channel_equalized, channel_name=channel_name)

    plt.show()


def img_comparison(image_rgb, result_image, mode):
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(result_image)
    plt.title(f'{mode} Equalized Image')
    plt.axis('off')


def histogram_comparison(image_rgb, result_image):
    # 显示原图直方图（RGB通道）
    plt.subplot(2, 2, 3)
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7)
    plt.title('Original RGB Histogram')
    plt.xlim([0, 256])

    # 显示处理后的直方图（RGB通道）
    plt.subplot(2, 2, 4)
    for i, color in enumerate(colors):
        hist = cv2.calcHist([result_image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7)
    plt.title('Equalized RGB Histogram')
    plt.xlim([0, 256])

    plt.tight_layout()


def equalized_channel_comparison(channel, channel_equalized, channel_name='Y Channel'):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(channel.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title(f'Original {channel_name} Histogram')

    plt.subplot(1, 2, 2)
    plt.hist(channel_equalized.flatten(), bins=256, range=(0, 256), color='orange', alpha=0.7)
    plt.title(f'Equalized {channel_name} Histogram')

    plt.tight_layout()
