import cv2
from hist_utils import histogram_equalization


def img_yuv_equalized(img_rgb):
    """
    将图像从 RGB 格式转换为 YUV 格式，然后再对 Y 通道(亮度)进行直方图均衡化，并且再返回 RGB 格式图像
    :param img_rgb: RGB 格式的图像
    :return: YUV 格式的图像
    """
    # 转换为 YUV 空间处理亮度通道
    # Y: 表示亮度，也就是灰阶值，控制图像的亮暗细节，包含了大部分可见的信息
    # U: 表示色度蓝色差分，记录图像中蓝色相对于亮度的差异，用来描述蓝色和黄色之间的对比
    # V: 表示色度红色差分，记录图像中红色相对于亮度的差异，用来描述红色和青色之间的对比
    yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)
    y_equalized = histogram_equalization(y)
    yuv_equalized = cv2.merge([y_equalized, u, v])

    # 再转换回 RGB 空间
    result_image = cv2.cvtColor(yuv_equalized, cv2.COLOR_YUV2RGB)

    return result_image, y, y_equalized


def img_hsv_equalized(img_rgb):
    """
    将图像从 RGB 格式转换为 HSV 格式，然后再对 V 通道(明度)进行直方图均衡化，并且再返回 RGB 格式图像
    :param img_rgb: RGB 格式的图像
    :return: HSV 格式的图像
    """
    # 转换为 HSV 空间处理明度通道
    # H: 表示色调，用于区分不同的颜色类别，取值范围通常为0°～360°
    # S: 表示饱和度，用于表示颜色的纯度或鲜艳程度，取值范围为0%～100%
    # V: 表示明度，描述颜色的亮度或明暗程度，取值范围为0%～100%
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v_equalized = histogram_equalization(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])

    # 再转换回 RGB 空间
    result_image = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2RGB)

    return result_image, v, v_equalized


def img_gray_equalized(img_rgb):
    """
    将图像从 RGB 格式转换为灰度图，然后再对灰度图进行直方图均衡化，并且再返回 RGB 格式图像
    :param img_rgb: RGB 格式的图像
    :return: 灰度图
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_equalized = histogram_equalization(gray)

    # 再转换回 RGB 空间
    result_image = cv2.cvtColor(gray_equalized, cv2.COLOR_GRAY2RGB)

    return result_image, gray, gray_equalized


def img_equalized(img_rgb, mode='YUV'):
    """
    三种直方图均衡化模式
    :param img_rgb: RGB 格式的图像
    :param mode: 模式
    :return: 均衡化结果
    """
    if mode is 'YUV':
        result_image, channel, channel_equalized = img_yuv_equalized(img_rgb=img_rgb)
    elif mode is 'HSV':
        result_image, channel, channel_equalized = img_hsv_equalized(img_rgb=img_rgb)
    elif mode is 'GRAY':
        result_image, channel, channel_equalized = img_gray_equalized(img_rgb=img_rgb)
    else:
        raise ValueError("Invalid mode.")

    return result_image, channel, channel_equalized
