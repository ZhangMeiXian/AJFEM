# -*- coding: utf-8 -*-

"""
@author:zhangmeixian
@time: 2021-10-22 12:00:00
"""
import sys


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
