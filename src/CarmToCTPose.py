# -*- coding: UTF-8 -*-
'''
@Project ：GeneratorLabelData 
@File    ：CarmToCTPose.py
@IDE     ：PyCharm 
@Author  ：soldier Hou
@E-mail  : 17853538105@163.com
@Date    ：2023/8/31 11:14 
'''

import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class CarmToCT_Pose:
    def __init__(self,pose):
        self.pose = pose

    def transform_mat(self):
        six_dof = self.pose

        # 将前三个旋转自由度转换为旋转矩阵
        r = R.from_euler('zyx', six_dof[:3], degrees=False)
        rotation_matrix = r.as_matrix()
        rotation_matrix[[1, 2]] = rotation_matrix[[2, 1]]
        # 添加平移分量
        translation = np.array(six_dof[3:])
        # 构建4x3变换矩阵
        transformation_matrix = np.hstack((rotation_matrix, translation.reshape(-1, 1)))
        return transformation_matrix