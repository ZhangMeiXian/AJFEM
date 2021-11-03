# -*- coding: utf-8 -*-

"""
@author:zhangmeixian
@time: 2021-10-22 12:00:00
"""

import librosa
from librosa import display
import os
import matplotlib.pyplot as plt


class DataExtractor:
    """
    基于原始音频提取FilterBank、LogMel
    """

    def __init__(self):
        # 原始音频所在文件夹（4Q数据集）
        self.parent_path = "data/4Q dataset 2018/MER_audio_taffc_dataset"

    def PreProcessor(self):
        """
        提取FilterBank、MFCC数据并存储
        :return: None
        """
        path = os.listdir(self.parent_path)
        for dir in path:
            # 四个象限音频所在文件夹
            class_path = "/".join([self.parent_path, dir])
            file_path = os.listdir(class_path)
            for file in file_path:
                # 音频文件路径
                audio_file = "/".join([class_path, file])
                filter_save_path = "/".join(["data/4Q dataset 2018/FilterBank", dir])
                logmel_save_path = "/".join(["data/4Q dataset 2018/LogMel", dir])
                if not os.path.exists(filter_save_path):
                    os.mkdir(filter_save_path)
                if not os.path.exists(logmel_save_path):
                    os.mkdir(logmel_save_path)
                spec_name = audio_file.split("/")[-1].split(".")[0] + ".png"

                # 提取原始音频，采样频率为原始频率sr=None
                y, sr = librosa.load(audio_file, sr=None)
                filter_save_path = filter_save_path + "/" + spec_name
                logmel_save_path = logmel_save_path + "/" + spec_name
                # 绘制filterbank图
                y_melf = librosa.filters.mel(sr=sr, n_fft=2048)
                plt.figure(figsize=(5, 5))
                display.specshow(y_melf)
                # display.specshow(y_melf, x_axis="time")
                # plt.ylabel("Mel filter bank")
                # plt.colorbar()
                plt.axis("off")
                plt.savefig(filter_save_path, bbox_inches='tight', pad_inches=0.0)
                # plt.savefig("filterbank", dpi=600, bbox_inches='tight')
                # plt.show()
                plt.clf()
                plt.close()

                # 绘制log-mel图
                melspec = librosa.feature.melspectrogram(y, sr)
                plt.figure(figsize=(5, 5))
                log_melspec = librosa.amplitude_to_db(melspec)
                display.specshow(log_melspec)
                # display.specshow(log_melspec, x_axis="time")
                plt.axis("off")
                plt.savefig(logmel_save_path, bbox_inches='tight', pad_inches=0.0)
                # plt.ylabel("Log mel")
                # plt.colorbar()
                # plt.savefig("logmel", dpi=600, bbox_inches='tight')
                # plt.show()


                plt.clf()
                plt.close()
                print("One audio file completed...")

                # break

        print("finished!")


if __name__ == "__main__":
    FBE = DataExtractor()
    FBE.PreProcessor()
