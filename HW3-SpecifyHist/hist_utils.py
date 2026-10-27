import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_cdf(hist):
    """计算直方图的累积分布函数 (CDF)"""
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-10)
    return cdf_normalized.astype(np.uint8)


def histogram_matching(source_gray, target_hist):
    """直方图规定化函数（单通道灰度图）"""
    source_hist = cv2.calcHist([source_gray], [0], None, [256], [0, 256])
    source_cdf = compute_cdf(source_hist)
    target_cdf = compute_cdf(target_hist)

    # 计算映射表
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        idx = np.argmin(np.abs(target_cdf - source_cdf[i]))
        mapping[i] = idx
    return mapping[source_gray]


def result_visualization(test_gray, target_gray, result_gray):
    """统一可视化：6个子图布局"""
    plt.figure(figsize=(15, 8))

    # 第一行：图像展示
    plt.subplot(2, 3, 1)
    plt.imshow(test_gray, cmap='gray')
    plt.title('Test Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(target_gray, cmap='gray')
    plt.title('Target Image')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(result_gray, cmap='gray')
    plt.title('Matched Result')
    plt.axis('off')

    # 第二行：直方图展示
    plt.subplot(2, 3, 4)
    plt.hist(test_gray.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title('Test Histogram')

    plt.subplot(2, 3, 5)
    plt.hist(target_gray.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
    plt.title('Target Histogram')

    plt.subplot(2, 3, 6)
    plt.hist(result_gray.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
    plt.title('Result Histogram')

    plt.tight_layout()
    plt.show()
