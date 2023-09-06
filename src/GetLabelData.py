# -*- coding: UTF-8 -*-
'''
@Project ：GeneratorLabelData 
@File    ：GetLabelData.py
@IDE     ：PyCharm 
@Author  ：soldier Hou
@E-mail  : 17853538105@163.com
@Date    ：2023/8/31 11:17 
'''
import SimpleITK as sitk
from configuration import Configuration
from generatorDRR import OptiBeamDRR
import numpy as np
from itertools import product
import cv2
import os
class DataLoader():
    def __init__(self, ct_path, config):
        self.__ct_path = ct_path
        self.__config = config
    def GetOriginVolume(self):
        vol = sitk.ReadImage(self.__ct_path)
        vol_spacing = vol.GetSpacing()
        self.__config["load_volume"]["resampled_spacing"] = vol_spacing  # 采样后spacing登记到配置文件
        vol_data = sitk.GetArrayFromImage(vol).transpose(2, 1, 0)
        # vol_data = np.rot90(vol_data, 1)
        return vol_data

    def GetVolume(self):
        resample_size = self.__config["load_volume"]["resampling_size"]

        vol = sitk.ReadImage(self.__ct_path)
        # 下采样防止显存溢出
        vol = self.__resampleVolumeWithSize(
            vol,
            [
                resample_size,
                int(vol.GetWidth() * resample_size / vol.GetHeight()),
                int(vol.GetDepth() * resample_size / vol.GetHeight()),
            ],
        )
        # size不变各个方向重采样到相同spacing
        vol_spacing = max(vol.GetSpacing())
        vol = self.__resampleVolumeWithSpacing(vol, [vol_spacing] * 3)
        vol_size = max(vol.GetSize())
        self.__config["load_volume"][
            "resampled_spacing"
        ] = vol_spacing  # 采样后spacing登记到配置文件
        self.__config["load_volume"]["resampled_size"] = vol_size  # 采样后size登记到配置文件
        vol_data = sitk.GetArrayFromImage(vol).transpose(2, 1, 0)

        # Rotation 90 degrees for making an AP view projectio
        if self.__config["drr_generator_type"] == "PSTGenerator":
            vol_data = np.rot90(vol_data, -1)  # rot90 将矩阵img逆时针旋转90°,负数时为顺时针


        # vol_data_temp = vol_data.copy()
        # vol_data = conv_hu_to_density(vol_data)
        # vol_data = vol_data * (vol_data_temp > 0)

        vol_data = np.flip(vol_data, axis=1)

        if self.__config["load_volume"]["flip"]:
            vol_data = np.flip(vol_data, axis=2)

        return vol_data

    def __resampleVolumeWithSize(self, vol, outsize):
        inputspacing = 0
        inputsize = 0
        inputorigin = [0, 0, 0]
        inputdir = [0, 0, 0]
        outspacing = [0, 0, 0]
        # 读取文件的size和spacing信息

        inputsize = vol.GetSize()
        inputspacing = vol.GetSpacing()

        transform = sitk.Transform()
        transform.SetIdentity()
        # 计算改变spacing后的size，用物理尺寸/体素的大小
        outspacing[0] = inputsize[0] * inputspacing[0] / outsize[0]
        outspacing[1] = inputsize[1] * inputspacing[1] / outsize[1]
        outspacing[2] = inputsize[2] * inputspacing[2] / outsize[2]

        # 设定重采样的一些参数
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputOrigin(vol.GetOrigin())
        resampler.SetOutputSpacing(outspacing)
        resampler.SetOutputDirection(vol.GetDirection())
        resampler.SetSize(outsize)
        newvol = resampler.Execute(vol)

        return newvol

    def __resampleVolumeWithSpacing(self, vol, outspacing):
        outsize = [0, 0, 0]
        inputspacing = 0
        inputsize = 0
        inputorigin = [0, 0, 0]
        inputdir = [0, 0, 0]
        # 读取文件的size和spacing信息

        inputsize = vol.GetSize()
        inputspacing = vol.GetSpacing()

        transform = sitk.Transform()
        transform.SetIdentity()
        # 计算改变spacing后的size，用物理尺寸/体素的大小
        outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
        outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
        outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)

        # 设定重采样的一些参数
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputOrigin(vol.GetOrigin())
        resampler.SetOutputSpacing(outspacing)
        resampler.SetOutputDirection(vol.GetDirection())
        resampler.SetSize(outsize)
        newvol = resampler.Execute(vol)

        return newvol

def getPath(config):
    path_CT = config["CTPath"]["path_CT_origin"]
    path_roi = config["CTPath"]["path_roi"]
    path = [path_CT,path_roi]
    return path

def getConfig(config_path):
    configuration = Configuration(config_path)
    config = configuration.GetConfig()
    return config

def get_search_space(init_pose, pyramid):
        (
            [rx_min, rx_max, rx_step],
            [ry_min, ry_max, ry_step],
            [rz_min, rz_max, rz_step],
            [dx_min, dx_max, dx_step],
            [dy_min, dy_max, dy_step],
            [dz_min, dz_max, dz_step],
        ) = pyramid

        rx_range = np.arange(rx_min, rx_max, rx_step) + init_pose[0]
        ry_range = np.arange(ry_min, ry_max, ry_step) + init_pose[1]
        rz_range = np.arange(rz_min, rz_max, rz_step) + init_pose[2]
        dx_range = np.arange(dx_min, dx_max, dx_step) + init_pose[3]
        dy_range = np.arange(dy_min, dy_max, dy_step) + init_pose[4]
        dz_range = np.arange(dz_min, dz_max, dz_step) + init_pose[5]

        poses = list(
            product(rx_range, ry_range, rz_range, dx_range, dy_range, dz_range)
        )
        return poses

if __name__ == "__main__":

    config_path = r'../config.json'
    config = getConfig(config_path)
    path = getPath(config)

    init_pose = [0, 0, 0, 0, 0, 0]
    pyramid = [
        [0, 0.1, 0.05],  # rx
        [0, 0.1, 0.05],  # ry
        [0, 0.1, 0.05],  # rz
        [-50, 50, 25],  # dx
        [-800, -400, 50],  # dz
        [-50, 50, 25],  # dy
    ]

    # pyramid = [
    #     [0, 1, 1],  # rx
    #     [0, 1, 1],  # ry
    #     [0, 1, 1],  # rz
    #     [0, 1, 1],  # dx
    #     [0, 2, 1],  # dz
    #     [0, 2, 1],  # dy
    #     # [-150, 350, 50],  # dz
    # ]

    search_space = get_search_space(init_pose,pyramid)

    save_path = "../data/spine+clavicle"
    for path_ in path:
        dataloader = DataLoader(path_, config)
        vol = dataloader.GetOriginVolume()
        drr_generator = OptiBeamDRR(vol, config)

        for pose in search_space:

            drr = drr_generator.GetDRR(path_, pose)

            pose_str = '_'.join(map(str, pose))
            folder_path = os.path.join(save_path, pose_str)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if (path_ == path[0]):
                drr = (drr * 255).astype(np.uint8)
                cv2.imwrite(folder_path + "/DRR.png", drr)

            else:
                drr[drr > 0] = 255
                drr.astype(np.uint8)
                cv2.imwrite(folder_path + "/mask.png", drr)






