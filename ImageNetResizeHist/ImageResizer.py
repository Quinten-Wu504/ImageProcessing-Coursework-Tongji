import os
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


class ImageResizer:
    def __init__(self, target_height, target_width, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.target_height = target_height
        self.target_width = target_width
        self.device = device

    def _resize_nearest(self, image):
        C, H, W = image.shape
        image = image.to(self.device)

        # 生成目标网格坐标
        row_idx = torch.linspace(0, H - 1, self.target_height, device=self.device)
        col_idx = torch.linspace(0, W - 1, self.target_width, device=self.device)
        row_idx = torch.round(row_idx).long()
        col_idx = torch.round(col_idx).long()

        grid_y, grid_x = torch.meshgrid(row_idx, col_idx)
        resized_images = image[:, grid_y, grid_x]

        return resized_images

    def _resize_bilinear(self, image):
        C, H, W = image.shape
        image = image.to(self.device)

        scale_y = H / self.target_height
        scale_x = W / self.target_width

        y = torch.arange(0, self.target_height, device=self.device).float() + 0.5
        x = torch.arange(0, self.target_width, device=self.device).float() + 0.5
        y = y * scale_y - 0.5
        x = x * scale_x - 0.5
        yy, xx = torch.meshgrid(y, x)

        y0 = torch.floor(yy).long().clamp(0, H - 1)
        x0 = torch.floor(xx).long().clamp(0, W - 1)
        y1 = (y0 + 1).clamp(0, H - 1)
        x1 = (x0 + 1).clamp(0, W - 1)

        dy = yy - y0.float()
        dx = xx - x0.float()

        wa = (1 - dy) * (1 - dx)
        wb = (1 - dy) * dx
        wc = dy * (1 - dx)
        wd = dy * dx

        wa = wa.unsqueeze(0)
        wb = wb.unsqueeze(0)
        wc = wc.unsqueeze(0)
        wd = wd.unsqueeze(0)

        Ia = image[:, y0, x0]
        Ib = image[:, y0, x1]
        Ic = image[:, y1, x0]
        Id = image[:, y1, x1]

        resized_images = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return resized_images

    def resize(self, images, mode='nearest'):
        if mode == 'nearest':
            return self._resize_nearest(images)
        elif mode == 'bilinear':
            return self._resize_bilinear(images)
        else:
            raise ValueError("Unsupported resize mode. Use 'nearest' or 'bilinear'.")


def resize_all_images(source_dir="./dataset/ImageNet2000", target_dir="./dataset/ImageNet2000_Resized", height=256, width=256, mode="nearest", device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(target_dir, exist_ok=True)
    transform = T.ToTensor()
    file_list = sorted([f for f in os.listdir(source_dir) if f.endswith(".JPEG")])

    resizer = ImageResizer(height, width, device)

    for i in tqdm(range(0, len(file_list)), desc="正在resize并保存"):
        fname = file_list[i]
        img_path = os.path.join(source_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).to("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = resizer.resize(img_tensor, mode=mode)

        # 保存每张图像
        img_pil = T.ToPILImage()(img_tensor.detach().cpu().clamp(0, 1))
        save_name = fname
        img_pil.save(os.path.join(target_dir, save_name))


if __name__ == "__main__":
    src_folder = "./dataset/ImageNet2000"
    dst_folder = "./dataset/ImageNet2000_Resized"

    # 获取用户输入的新尺寸
    target_h = 256
    target_w = 256

    resize_all_images(
        source_dir=src_folder,
        target_dir=dst_folder,
        height=target_h,
        width=target_w,
        mode='nearest'
    )

    print("\n所有图像已成功resize并保存到:", dst_folder)
