import numpy as np
import cv2
import matplotlib.pyplot as plt


def ideal_filter(shape, cutoff, highpass=False):
    rows, cols = shape
    center = (rows // 2, cols // 2)
    H = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center[0]) ** 2 + (v - center[1]) ** 2)
            if (D <= cutoff and not highpass) or (D > cutoff and highpass):
                H[u, v] = 1
    return H


def butterworth_filter(shape, cutoff, order=2, highpass=False):
    rows, cols = shape
    center = (rows // 2, cols // 2)
    H = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center[0]) ** 2 + (v - center[1]) ** 2)
            H[u, v] = 1 / (1 + (D / cutoff) ** (2 * order))
    return 1 - H if highpass else H


def gaussian_filter(shape, cutoff, highpass=False):
    rows, cols = shape
    center = (rows // 2, cols // 2)
    H = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - center[0]) ** 2 + (v - center[1]) ** 2)
            H[u, v] = np.exp(- (D ** 2) / (2 * (cutoff ** 2)))
    return 1 - H if highpass else H


def apply_filter(image, filter_func, cutoff, highpass=False, order=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)

    if filter_func == 'ideal':
        H = ideal_filter(gray.shape, cutoff, highpass)
    elif filter_func == 'butterworth':
        H = butterworth_filter(gray.shape, cutoff, order, highpass)
    elif filter_func == 'gaussian':
        H = gaussian_filter(gray.shape, cutoff, highpass)

    filtered_dft = dft_shift * H
    idft_shift = np.fft.ifftshift(filtered_dft)
    filtered_img = np.fft.ifft2(idft_shift)
    filtered_img = np.abs(filtered_img)
    return filtered_img, H


# 读取图像
image = cv2.imread('./TestData/cat_test.jpg')
cutoff_90, cutoff_60 = 50, 30  # 假设的截止频率

# 低通滤波
lowpass_90, H_90 = apply_filter(image, 'gaussian', cutoff_90, highpass=False)
lowpass_60, H_60 = apply_filter(image, 'gaussian', cutoff_60, highpass=False)

# 高通滤波
highpass_90, H_90_high = apply_filter(image, 'gaussian', cutoff_90, highpass=True)
highpass_60, H_60_high = apply_filter(image, 'gaussian', cutoff_60, highpass=True)

# 显示结果
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), ax[0, 0].set_title("Original Image")
ax[0, 1].imshow(lowpass_90, cmap='gray'), ax[0, 1].set_title("Lowpass 90% Energy")
ax[0, 2].imshow(lowpass_60, cmap='gray'), ax[0, 2].set_title("Lowpass 60% Energy")
ax[1, 1].imshow(highpass_90, cmap='gray'), ax[1, 1].set_title("Highpass 90% Energy")
ax[1, 2].imshow(highpass_60, cmap='gray'), ax[1, 2].set_title("Highpass 60% Energy")
plt.show()
