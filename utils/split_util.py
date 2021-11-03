# -*- coding: utf-8 -*-

"""
@author:zhangmeixian
@time: 2021-10-22 12:00:00
"""

import os
import shutil


class SplitData:
    """
    划分数据，将数据转换成训练集和测试集
    """

    def __init__(self):
        pass

    def split_data(self, dir, train_size=0.8):
        """
        将数据集划分成训练集和测试集并分开存储到不同的文件夹
        :param dir:
        :return:
        """
        parent_path = os.listdir(dir)
        spec_cls = dir.split("/")[-1]
        for cls in parent_path:
            samples_path = "/".join([dir, cls])
            samples = os.listdir(samples_path)
            count = 0
            train_count = int(len(samples) * train_size)
            for sample in samples:
                sample_path = "/".join([samples_path, sample])
                if sample.endswith(".png"):
                    count += 1
                    if count <= train_count:
                        new_path = "/Users/zhangmeixian/Desktop/Python/MER/AJFEM/data/dataset/train_set_{}"\
                            .format(spec_cls)
                        if not os.path.exists(new_path):
                            os.mkdir(new_path)
                        desc = os.path.join(os.path.abspath(new_path),
                                            sample.split(".")[0] + "_" + cls + ".png")
                        shutil.move(sample_path, desc)
                    else:
                        new_path = "/Users/zhangmeixian/Desktop/Python/MER/AJFEM/data/dataset/test_set_{}"\
                            .format(spec_cls)
                        if not os.path.exists(new_path):
                            os.mkdir(new_path)
                        desc = os.path.join(os.path.abspath(new_path),
                                            sample.split(".")[0] + "_" + cls + ".png")
                        shutil.move(sample_path, desc)


if __name__=="__main__":
    SD = SplitData()
    SD.split_data("/Users/zhangmeixian/Desktop/Python/MER/AJFEM/data/4Q dataset 2018/LogMel")
    SD.split_data("/Users/zhangmeixian/Desktop/Python/MER/AJFEM/data/4Q dataset 2018/FilterBank")

