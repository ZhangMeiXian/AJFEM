# -*- coding: utf-8 -*-

"""
@author:zhangmeixian
@time: 2021-10-22 12:00:00
"""
import torch.nn as nn
import torch


class CnnGruSvm(torch.nn.Module):
    """
    two-branch CNN + GRU-SVM
    """

    def __init__(self):
        super(CnnGruSvm, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 1)
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv3_1 = nn.Conv2d(128, 256, 1)
        self.avgpool1 = nn.AvgPool2d(2, 2)

        self.conv1_2 = nn.Conv2d(3, 64, 1)
        self.conv2_2 = nn.Conv2d(64, 128, 3)
        self.conv3_2 = nn.Conv2d(128, 256, 1)
        self.avgpool2 = nn.AvgPool2d(2, 2)











