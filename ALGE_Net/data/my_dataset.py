#训练
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import pydicom as dicom


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, data_dir, batch_size, transform=None):
        self.images_path = images_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.prepro = self.preprocess

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        index = index % len(self.images_path)
        # 根据索引提取单行数据
        row = self.images_path.iloc[index]

        # 获取DICOM文件路径和ROI坐标以及孕周信息
        dicom_path = row.iloc[1]  # 第2列是DICOM文件路径
        left = row.iloc[2]  # 第3列是左下角的x坐标
        bottom = row.iloc[3]  # 第4列是左下角的y坐标
        right = row.iloc[4]  # 第5列是右上角的x坐标
        top = row.iloc[5]  # 第6列是右上角的y坐标
        week = row.iloc[6]  # 假设第7列是孕周信息，用作标签

        # 读取DICOM文件并提取ROI
        dicom_file = dicom.dcmread(dicom_path)
        image = dicom_file.pixel_array
        image_normalized = (image - image.min()) / (image.max() - image.min())
        image_8bit = (image_normalized * 255).astype(np.uint8)
        roi = image_8bit[top:bottom, left:right]

        # 将ROI转换为PIL Image，并转换为灰度图
        img = Image.fromarray(roi).convert('L')

        # 应用ToTensor转换
        transf = transforms.ToTensor()
        img = transf(img)

        # 应用额外的转换（如果有）
        images = []
        images.append(img)
        img = self.prepro(images)
        img = torch.squeeze(img, dim=0)

        if self.transform is not None:
            img = self.transform(img)
        # 创建标签
        label = np.asarray([row['day']]).astype('float32')

        return img, label

    class collater():
        def __init__(self, *params):
            self.params = params

        def __call__(self, batch):
            images, labels = tuple(zip(*batch))
            img = torch.stack(images, dim=0)

            labels = torch.as_tensor(np.array(labels))
            return img, labels

    # 预处理 padding
    def preprocess(self, images):
        """
                    Args:
                      batch_inputs: 图片张量列表
                    Return:
                      padded_images: 填充后的批量图片张量
                      image_sizes_orig: 原始图片尺寸信息
                """
        ## 保留原始图片尺寸
        image_sizes_orig = [[image.shape[-2], image.shape[-1]] for image in images]
        ## 找到最大尺寸
        max_size = max([max(image_size[0], image_size[1]) for image_size in image_sizes_orig])

        ## 构造批量形状 (batch_size, channel, max_size, max_size)
        batch_shape = (len(images), images[0].shape[0], max_size, max_size)

        padded_images = images[0].new_full(batch_shape, 0.0)
        for padded_img, img in zip(padded_images, images):
            c, h, w = img.shape[:3]
            padded_img[..., :h, :w].copy_(img)

        return padded_images