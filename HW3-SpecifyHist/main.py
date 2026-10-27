import cv2
from hist_utils import histogram_matching, result_visualization


def test_histogram_matching():
    # 加载测试图像和目标图像（统一转为灰度图）
    test_img = cv2.imread('./TestData/test_image.jpg')
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    target_img = cv2.imread('./TestData/target_image.jpg')
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    target_hist = cv2.calcHist([target_gray], [0], None, [256], [0, 256])

    # 直方图规定化
    result_gray = histogram_matching(test_gray, target_hist)

    # 可视化
    result_visualization(
        test_gray=test_gray,
        target_gray=target_gray,
        result_gray=result_gray
    )


if __name__ == '__main__':
    test_histogram_matching()
