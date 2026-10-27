import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_operations import *


def load_image(path):
    """加载图像并转为灰度图"""
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def plot_results(images, titles, rows, cols):
    """绘制多子图结果"""
    plt.figure(figsize=(15, 8))
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def test_logical_operations():
    # 加载测试图像
    original = load_image('./TestData/test_plane.jpg')
    h, w = original.shape

    # 生成带圆形的掩膜（二值图）
    circle_mask = generate_circle_mask((h, w), center=(w // 2, h // 2), radius=200)

    # 将原图二值化（可选，若需要与原图进行逻辑运算）
    _, binary_original = cv2.threshold(original, 128, 255, cv2.THRESH_BINARY)

    # 执行逻辑运算
    and_result = logical_operations(binary_original, circle_mask, 'AND')
    or_result = logical_operations(binary_original, circle_mask, 'OR')
    xor_result = logical_operations(binary_original, circle_mask, 'XOR')
    not_result = logical_operations(binary_original, None, 'NOT')
    print('Logical Operation Has Finished!')

    # 滤波测试
    mean_filtered = mean_filter(original, ksize=10)
    gaussian_filtered = gaussian_filter(original, ksize=10, sigma=1.5)
    print('Filter Operation Has Finished!')

    # 边缘检测测试
    sobel_edges = sobel_edge_detection(original)
    prewitt_edges = prewitt_edge_detection(original)
    laplacian_sharp = laplacian_sharpening(original)
    print('Edge Detection Has Finished!')

    # 可视化结果
    images = [
        original,
        circle_mask,
        binary_original,
        and_result,
        or_result,
        xor_result,
        not_result,
        mean_filtered,
        gaussian_filtered,
        sobel_edges,
        prewitt_edges,
        laplacian_sharp,
    ]
    titles = [
        'Original Image',
        'Generated Circle Mask',
        'Binary Original',
        'AND Operation',
        'OR Operation',
        'XOR Operation',
        'NOT Operation (Image Inverted)',
        'Mean Filter',
        'Gaussian Filter',
        'Sobel',
        'Prewitt',
        'Laplacian Sharp',
    ]
    plot_results(images, titles, rows=3, cols=4)


if __name__ == '__main__':
    test_logical_operations()
