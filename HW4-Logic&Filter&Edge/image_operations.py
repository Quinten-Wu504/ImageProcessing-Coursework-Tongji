import cv2
import numpy as np


def generate_circle_mask(shape, center=None, radius=200):
    """
    生成带有圆形的二值掩膜
    :param shape: 目标图像的形状 (height, width)
    :param center: 圆心坐标 (x, y)，若为 None 则随机生成
    :param radius: 圆半径
    :return: 二值掩膜（0 和 255）
    """
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    # 随机生成圆心位置（若未指定）
    if center is None:
        cx = np.random.randint(radius, w - radius)
        cy = np.random.randint(radius, h - radius)
    else:
        cx, cy = center
    # 绘制圆形
    cv2.circle(mask, (cx, cy), radius, color=255, thickness=-1)
    return mask


def logical_operations(img1, img2, operation='AND'):
    """图像逻辑运算（要求输入为二值图）"""
    if operation == 'AND':
        return cv2.bitwise_and(img1, img2)
    elif operation == 'OR':
        return cv2.bitwise_or(img1, img2)
    elif operation == 'XOR':
        return cv2.bitwise_xor(img1, img2)
    elif operation == 'NOT':
        return cv2.bitwise_not(img1)
    else:
        raise ValueError("Unsupported logical operation.")


def apply_filter(image, kernel):
    """通用卷积滤波（支持自定义核）"""
    ksize = kernel.shape[0]
    pad = ksize // 2
    # 边界填充（镜像）
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    output = np.zeros_like(image, dtype=np.float32)
    # 滑动窗口卷积
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            output[y, x] = np.sum(padded[y:y+ksize, x:x+ksize] * kernel)
    return np.clip(output, 0, 255).astype(np.uint8)


def mean_filter(image, ksize=3):
    """均值滤波"""
    kernel = np.ones((ksize, ksize)) / (ksize**2)
    return apply_filter(image, kernel)


def gaussian_filter(image, ksize=3, sigma=1.0):
    """高斯滤波"""
    ax = np.linspace(-(ksize//2), ksize//2, ksize)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return apply_filter(image, kernel)


def sobel_edge_detection(image):
    """Sobel 边缘检测"""
    sobel_x = np.array([[-1,  0,  1],
                                 [-2,  0,  2],
                                 [-1,  0,  1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=np.float32)
    grad_x = apply_filter(image, sobel_x)
    grad_y = apply_filter(image, sobel_y)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    return np.clip(gradient, 0, 255).astype(np.uint8)


def prewitt_edge_detection(image):
    """Prewitt 边缘检测"""
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    grad_x = apply_filter(image, prewitt_x)
    grad_y = apply_filter(image, prewitt_y)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    return np.clip(gradient, 0, 255).astype(np.uint8)


def laplacian_sharpening(image):
    """拉普拉斯锐化"""
    laplacian_kernel = np.array([[-1, -1, -1],
                                          [-1,  8, -1],
                                          [-1, -1, -1]], dtype=np.float32)
    sharpened = apply_filter(image, laplacian_kernel)
    sharpened = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    # 叠加原图增强边缘
    return np.clip(sharpened, 0, 255).astype(np.uint8)
