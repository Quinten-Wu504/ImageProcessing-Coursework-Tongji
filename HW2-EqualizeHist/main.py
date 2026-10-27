import cv2
from img_equalized import img_equalized
from hist_utils import result_visualization


def test_histogram(mode='YUV'):
    image = cv2.imread('./TestData/test_colorful.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result_image, channel, channel_equalized = img_equalized(img_rgb=image_rgb, mode=mode)
    result_visualization(
        image_rgb=image_rgb,
        result_image=result_image,
        channel=channel,
        channel_equalized=channel_equalized,
        mode=mode
    )


if __name__ == '__main__':
    print('RGB to YUV Histogram Equalization')
    test_histogram(mode='YUV')

    print('RGB to HSV Histogram Equalization')
    test_histogram(mode='HSV')

    print('RGB to GRAY Histogram Equalization')
    test_histogram(mode='GRAY')
