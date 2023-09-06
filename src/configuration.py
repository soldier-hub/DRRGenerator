# -*- coding: UTF-8 -*-
'''
@Project ：GeneratorLabelData 
@File    ：configuration.py
@IDE     ：PyCharm 
@Author  ：soldier Hou
@E-mail  : 17853538105@163.com
@Date    ：2023/8/31 11:31 
'''

import json
import os

class Configuration(object):
    def __init__(self, config_file_path):
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(
                f"config file not found at path: {config_file_path}"
            )
        else:
            with open(config_file_path) as f:
                configuration = json.load(f)
                self.config = configuration

    def GetConfig(self):
        return self.config
