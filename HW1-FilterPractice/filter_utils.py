import numpy as np
from scipy.optimize import bisect


def ideal_filter(img, energy_percent, filter_type):
    rows, cols = img.shape

    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 计算总能量
    total_energy = np.sum(np.abs(fshift) ** 2)

    # 生成距离矩阵
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
    distance = np.sqrt(x ** 2 + y ** 2)

    # 展平数据并根据滤波器类型排序
    flat_dist = distance.flatten()
    flat_magnitude_sqrt = (np.abs(fshift) ** 2).flatten()

    # 修改关键部分：高通滤波器需要反向累积能量（从高频到低频）
    if filter_type == 'low':
        # 低通：按距离从小到大累积
        sorted_indices = np.argsort(flat_dist)
    elif filter_type == 'high':
        # 高通：按距离从大到小累积
        sorted_indices = np.argsort(-flat_dist)  # 反向排序
    else:
        raise ValueError("Invalid filter type.")

    sorted_dist = flat_dist[sorted_indices]
    sorted_magnitude_sqrt = flat_magnitude_sqrt[sorted_indices]

    # 计算累积能量并找到截止点
    cumulative_energy = np.cumsum(sorted_magnitude_sqrt)
    # print(cumulative_energy / total_energy)
    cutoff_index = np.argmax(cumulative_energy >= (energy_percent / 100) * total_energy)
    cutoff_radius = sorted_dist[cutoff_index] if cutoff_index > 0 else 0

    # 构建滤波器
    if filter_type == 'low':
        filter_mask = (distance <= cutoff_radius).astype(np.uint8)
    elif filter_type == 'high':
        filter_mask = (distance >= cutoff_radius).astype(np.uint8)
    else:
        raise ValueError("Invalid filter type.")

    # 应用滤波器并逆变换
    fshift_filtered = fshift * filter_mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_filtered = np.abs(img_filtered)

    print(f"Cutoff radius of ideal filter for {filter_type} {energy_percent}%: {cutoff_radius}")

    return img_filtered, filter_mask


def gaussian_filter(img, energy_percent, filter_type='low'):
    rows, cols = img.shape

    # 傅里叶变换并中心化
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 计算总能量（幅度平方和）
    total_energy = np.sum(np.abs(fshift) ** 2)
    target_energy = (energy_percent / 100) * total_energy

    # 生成距离矩阵
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    distance = np.sqrt(x ** 2 + y ** 2)

    # 定义能量计算函数（用于二分查找）
    def energy_for_sigma(gaussian_sigma):
        if filter_type == 'low':
            gaussian_mask = np.exp(-(distance ** 2) / (2 * gaussian_sigma ** 2))
        elif filter_type == 'high':
            gaussian_mask = 1 - np.exp(-(distance ** 2) / (2 * gaussian_sigma ** 2))
        else:
            raise ValueError("Invalid filter type.")

        filtered_energy = np.sum(np.abs(fshift * gaussian_mask) ** 2)

        return filtered_energy - target_energy

    # 通过二分查找确定sigma
    # 修改代码中的sigma_min和sigma_max定义部分
    sigma_min = 1e-6
    if filter_type == 'low':
        sigma_max = 10 * max(rows, cols)
    elif filter_type == 'high':
        sigma_max = 10 * max(rows, cols)
    else:
        raise ValueError("Invalid filter type.")

    # 检查函数端点符号
    f_a = energy_for_sigma(sigma_min)
    f_b = energy_for_sigma(sigma_max)

    if f_a * f_b >= 0:
        # 提示用户调整参数或直接返回默认值
        if filter_type == 'low':
            sigma = 1e-10  # 返回一个合理的小sigma
        else:
            sigma = 10 * max(rows, cols)  # 返回一个较大的sigma
        print(f_a / total_energy, f_b / total_energy)
        print(f"Warning: Cannot find exact sigma for {energy_percent}% energy. Using heuristic value sigma={sigma}.")
    else:
        sigma = bisect(energy_for_sigma, sigma_min, sigma_max, xtol=1e-3)

    # 构建最终滤波器
    if filter_type == 'low':
        mask = np.exp(-(distance ** 2) / (2 * (np.abs(sigma) ** 2)))
    elif filter_type == 'high':
        mask = 1 - np.exp(-(distance ** 2) / (2 * (np.abs(sigma) ** 2)))
    else:
        raise ValueError("Invalid filter type.")

    # 应用滤波器并逆变换
    fshift_filtered = fshift * mask
    f_filtered = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.abs(np.fft.ifft2(f_filtered))
    return img_filtered, mask


def butterworth_filter(img, energy_percent, filter_type='low', n=2):
    rows, cols = img.shape

    # 傅里叶变换并中心化
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 计算总能量（幅度平方和）
    total_energy = np.sum(np.abs(fshift) ** 2)
    target_energy = (energy_percent / 100) * total_energy

    # 生成距离矩阵
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    distance = np.sqrt(x ** 2 + y ** 2)

    # 定义能量计算函数（用于二分查找）
    def energy_for_cutoff(butterworth_cutoff):
        if filter_type == 'low':
            butterworth_mask = 1 / (1 + (distance / butterworth_cutoff) ** (2 * n))
        elif filter_type == 'high':
            tmp_safe_distance = np.where(distance < 1e-10, 1e-10, distance)
            butterworth_mask = 1 / (1 + (butterworth_cutoff / tmp_safe_distance)**(2*n))
        else:
            raise ValueError("Invalid filter type.")

        butterworth_mask[np.isnan(butterworth_mask)] = 0  # 处理除零问题
        filtered_energy = np.sum(np.abs(fshift * butterworth_mask) ** 2)

        return filtered_energy - target_energy

    # 通过二分查找确定cutoff
    # 修改代码中的cutoff_min和cutoff_max定义部分
    cutoff_min = 1e-10
    if filter_type == 'low':
        cutoff_max = 10 * max(rows, cols)  # 扩大上限
    elif filter_type == 'high':
        cutoff_max = 10 * max(rows, cols)
    else:
        raise ValueError("Invalid filter type.")

    # 检查函数端点符号
    f_a = energy_for_cutoff(cutoff_min)
    f_b = energy_for_cutoff(cutoff_max)

    if f_a * f_b >= 0:
        # 提示用户调整参数或直接返回默认值
        if filter_type == 'low':
            cutoff = 1e-10  # 返回一个合理的小cutoff
        else:
            cutoff = 10 * max(rows, cols)  # 返回一个较大的cutoff
        print(f_a / total_energy, f_b / total_energy)
        print(f"Warning: Cannot find exact cutoff for {energy_percent}% energy. Using heuristic value cutoff={cutoff}.")
    else:
        cutoff = bisect(energy_for_cutoff, cutoff_min, cutoff_max, xtol=1e-3)

    # 构建最终滤波器
    if filter_type == 'low':
        mask = 1 / (1 + (distance / cutoff) ** (2 * n))
    elif filter_type == 'high':
        safe_distance = np.where(distance < 1e-10, 1e-10, distance)
        mask = 1 / (1 + (cutoff / safe_distance) ** (2 * n))
    else:
        raise ValueError("Invalid filter type.")
    mask[np.isnan(mask)] = 0

    # 应用滤波器并逆变换
    fshift_filtered = fshift * mask
    f_filtered = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.abs(np.fft.ifft2(f_filtered))
    return img_filtered, mask


if __name__ == '__main__':
    pass
