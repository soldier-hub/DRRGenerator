# -*- coding: UTF-8 -*-
'''
@Project ：GeneratorLabelData 
@File    ：ImgRay.py
@IDE     ：PyCharm 
@Author  ：soldier Hou
@E-mail  : 17853538105@163.com
@Date    ：2023/8/31 11:15 
'''

import numpy as np


class GetRay():
    def __init__(self, config) -> None:
        self.nOrder = config["carm"]["distortion_order"]  # Default to 4
        self.matA = np.array(config["carm"]["row_distortion"])
        # self.matA.reshape(self.nOrder+1, self.nOrder+1)
        self.matB = np.array(config["carm"]["column_distortion"])
        # self.matB.reshpe(self.nOrder+1, self.nOrder+1)
        self.use_distortion = config["carm"]["use_distortion"]

        self.FocalPix = np.array(config["carm"]["focalPix"])
        self.CenterPix = np.array(config["carm"]["centerPix"])

        self.nSxVid = config["DRR"]["size_x"]
        self.nSyVid = config["DRR"]["size_y"]

    def multiply_elementwise_and_assign(self, lhs):
        lhs = lhs * self.FocalPix
        return lhs

    def DenormalizePoint(self, Point):
        result = self.multiply_elementwise_and_assign(Point)
        result = result + self.CenterPix
        return result

    def apply_to_normalized_point(self, Point):
        # calculate powers of X and Y
        vPowX = [pow(Point[0], i) for i in range(self.nOrder + 1)]
        vPowY = [pow(Point[1], i) for i in range(self.nOrder + 1)]
        u, v = 0.0, 0.0
        for i in range(self.nOrder + 1):
            for j in range(i + 1):
                u += self.matA[i][j] * vPowX[i - j] * vPowY[j]
                v += self.matB[i][j] * vPowX[i - j] * vPowY[j]
        Point = np.array([u, v])
        return Point

    def divide_elementwise(self,rhs1):
        rhs1 = rhs1 - self.CenterPix
        lhs = rhs1 / self.FocalPix
        return lhs

    def NormalizePoint(self,Point):
        PointNorm = self.divide_elementwise(Point)
        return PointNorm

    def Apply(self,Point):
        if self.use_distortion:
            return self.DenormalizePoint(self.apply_to_normalized_point(self.NormalizePoint(Point)))
        else:
            return self.DenormalizePoint(self.NormalizePoint(Point))
    def UndistortPix(self, rhs):
        InMask_rhs = self.flip(self.Apply(self.flip(rhs)))
        return InMask_rhs

    def flip(self, rhs, hflip=True):
        width = self.nSxVid
        lhs = rhs.copy()
        if hflip:
            lhs[0] = width - 1 - lhs[0]
        return lhs

    def PixToNormalized(self, Distorted):
        Corr = self.UndistortPix(Distorted)
        CorrFlip = self.flip(Corr)
        x = (CorrFlip[0] - self.CenterPix[0]) / self.FocalPix[0]
        y = (CorrFlip[1] - self.CenterPix[1]) / self.FocalPix[1]
        point2D = np.array([x, y], dtype=float)
        return point2D

    def PixToRay(self,Point):
        pointZ0 = np.array([0, 0], dtype=float)
        pointZ1 = self.PixToNormalized(Point)
        return pointZ0, pointZ1


    def GetImgRay(self,):
        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('w', np.float32)])
        ImgRay = np.zeros([self.nSxVid, self.nSyVid], dtype=dtype)
        for j in range(self.nSyVid):
            for i in range(self.nSxVid):
                Pixel = np.array([i, j])
                pointA, pointB = self.PixToRay(Pixel)
                ImgRay[i, j] = (pointA[0], pointA[1], pointB[0], pointB[1])
        return ImgRay