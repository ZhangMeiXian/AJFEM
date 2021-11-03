# -*- coding: utf-8 -*-

"""
@author:zhangmeixian
@time: 2021-10-22 12:00:00
"""

from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd

CLASS_DIC = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}


class MyDataset(Dataset):
    """
    读取频谱数据集
    构建训练集和测试集
    需要分别存放在不同的文件夹中
    """
    def __init__(self, path_dir, transform=None):
        self.path_dir = path_dir
        self.transform = transform
        self.specs = os.listdir(self.path_dir)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, index):
        spec = self.specs[index]  # 根据索引获取图像文件名称
        cls = spec.split("_")[-1].split(".")[0]
        label = CLASS_DIC[cls]
        spec_path = os.path.join(self.path_dir, spec)  # 获取图像的路径或目录
        spec = Image.open(spec_path).convert('RGB')  # 读取图像

        if self.transform is not None:
            spec = self.transform(spec)

        # 返回对应频谱图和标签
        return spec, label



