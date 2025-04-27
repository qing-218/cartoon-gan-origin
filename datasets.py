import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import cv2
import os
import numpy as np
from .transforms import get_default_transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from numba import njit, jit, prange

# 快速循环优化
def fast_loop(gauss_img, pad_img, kernel_size, gauss, dilation):
    idx = np.where(dilation != 0)
    loops = int(np.sum(dilation != 0))
    for i in range(loops):
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(
            pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(
            pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(
            pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))
    return gauss_img

# 处理单张图片
def edge_job(args):
    output_size = (256, 256)
    path, gauss, img_size, kernel, kernel_size, save, n = args
    try:
        rgb_img = cv2.imread(path)
        gray_img = cv2.imread(path, 0)
        if rgb_img is None or gray_img is None:
            print(f"[Warning] 读取失败: {path}")
            return
        
        rgb_img = np.array(ImageOps.fit(Image.fromarray(rgb_img), img_size, Image.Resampling.LANCZOS))
        pad_img = np.pad(rgb_img, ((3, 3), (3, 3), (0, 0)), mode='reflect')
        gray_img = np.array(ImageOps.fit(Image.fromarray(gray_img), img_size, Image.Resampling.LANCZOS))

        edges = cv2.Canny(gray_img, 150, 500)
        dilation = cv2.dilate(edges, kernel)

        _gauss_img = np.copy(rgb_img)
        gauss_img = fast_loop(_gauss_img, pad_img, kernel_size, gauss, dilation)

        rgb_img = cv2.resize(rgb_img, output_size, interpolation=cv2.INTER_AREA)
        gauss_img = cv2.resize(gauss_img, output_size, interpolation=cv2.INTER_AREA)

        comb_img = np.concatenate((rgb_img, gauss_img), axis=1)
        out_path = os.path.join(save, f"{n}.jpg")
        cv2.imwrite(out_path, comb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    except Exception as e:
        print(f"[Error] 处理失败: {path}, 错误信息: {e}")

# 单线程版的边缘增强主函数
def edge_promoting(root, save):
    img_size = (384, 384)
    kernel_size = 5
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss @ gauss.T  # 高斯核

    pbar = tqdm.tqdm(total=len(file_list))

    for n, f in enumerate(file_list):
        path = os.path.join(root, f)
        args = (path, gauss, img_size, kernel, kernel_size, save, n)
        edge_job(args)
        pbar.update(1)

# -------------------
# 后面的 DataLoader 相关也整理好
# -------------------

class ImageDataset(Dataset):
    """Image dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.root_files = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.root_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.root_files[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def get_dataloader(path="./datasets/real_images", size=256, bs=64, trfs=None, flip=0.005):
    "If no transforms specified, use default transforms"
    if not trfs:
        trfs = get_default_transforms(size=size)
    dset = ImageDataset(path, transform=trfs)
    return DataLoader(dset, batch_size=bs, num_workers=4, drop_last=True, shuffle=True)
