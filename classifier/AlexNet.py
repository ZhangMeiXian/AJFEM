# -*- coding: utf-8 -*-

"""
@author:zhangmeixian
@time: 2021-10-22 12:00:00
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class AlexNet(torch.nn.Module):
    """
    single log-mel CNN + LSTM
    """

    def __init__(self, num_classes=4, n_hidden=256, num_layers=2, init_weights=False):
        super(SingleCnnLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.blstm = nn.LSTM(input_size=128 * 6, hidden_size=n_hidden,
                             num_layers=num_layers, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(126 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # # LSTM
        # x = x.view(-1, x.shape[0], x.shape[1] * x.shape[2])
        # # print(x.shape)
        # output, (h_n, c_n) = self.blstm(x)
        # # print(output.shape)
        # # 将正向lstm和反向lstm的最后一层隐藏层结果进行拼接
        # encoding1 = torch.cat([h_n[0], h_n[1]], dim=1)
        # # 将正向lstm和反向lstm的输出结果进行拼接
        # encoding2 = torch.cat([output[0], output[1]], dim=1)
        out = torch.flatten(x, start_dim=1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



