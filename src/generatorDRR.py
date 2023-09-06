# -*- coding: UTF-8 -*-
'''
@Project ：GeneratorLabelData 
@File    ：generatorDRR.py
@IDE     ：PyCharm 
@Author  ：soldier Hou
@E-mail  : 17853538105@163.com
@Date    ：2023/8/31 11:12 
'''



import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from scipy.ndimage import zoom
from CarmToCTPose import CarmToCT_Pose
from ImgRay import GetRay

import torch
import matplotlib.pyplot as plt
import cv2


class OptiBeamDRR:
    def __init__(self, vol, config) -> None:
        self.nSxVid = config["DRR"]["size_x"]
        self.nSyVid = config["DRR"]["size_y"]

        ct_vol = np.ascontiguousarray(vol.astype(np.float32))
        self.d_data = cuda.mem_alloc(ct_vol.nbytes)
        # Copy data to GPU memory
        cuda.memcpy_htod(self.d_data, ct_vol)

        self.ct_spacing = config["load_volume"]["resampled_spacing"]
        self.ct_dim = ct_vol.shape
        self.src_det = config["carm"]["src_det"]

        self.__device = config["PSTGenerator"]["device"]

        ImgRay = GetRay(config).GetImgRay()
        self.ImgRay_gpu = cuda.mem_alloc(ImgRay.nbytes)
        cuda.memcpy_htod(self.ImgRay_gpu, ImgRay)

        self.DRR = torch.empty([self.nSxVid, self.nSyVid], dtype=torch.float32, device=self.__device)

        self.mod = SourceModule("""

           __device__ float trilinear_interpolation(float u, float v, float w,
                                                    float* data, int dim_z, int dim_y, int dim_x) {
                int floor_u = int(u);
                int floor_v = int(v);
                int floor_w = int(w);

                int ceil_u = floor_u + 1;
                int ceil_v = floor_v + 1;
                int ceil_w = floor_w + 1;

                float frac_u = u - floor_u;
                float frac_v = v - floor_v;
                float frac_w = w - floor_w;

                int index_c000 = floor_w + dim_z * (floor_v + dim_y * floor_u);
                int index_c001 = floor_w + dim_z * (floor_v + dim_y * ceil_u);
                int index_c010 = floor_w + dim_z * (ceil_v + dim_y * floor_u);
                int index_c011 = floor_w + dim_z * (ceil_v + dim_y * ceil_u);
                int index_c100 = ceil_w + dim_z * (floor_v + dim_y * floor_u);
                int index_c101 = ceil_w + dim_z * (floor_v + dim_y * ceil_u);
                int index_c110 = ceil_w + dim_z * (ceil_v + dim_y * floor_u);
                int index_c111 = ceil_w + dim_z * (ceil_v + dim_y * ceil_u);

                float c000 = data[index_c000];
                float c100 = data[index_c001];
                float c010 = data[index_c010];
                float c110 = data[index_c011];
                float c001 = data[index_c100];
                float c101 = data[index_c101];
                float c011 = data[index_c110];
                float c111 = data[index_c111];

                float c00 = c000 * (1 - frac_u) + c100 * frac_u;
                float c01 = c001 * (1 - frac_u) + c101 * frac_u;
                float c10 = c010 * (1 - frac_u) + c110 * frac_u;
                float c11 = c011 * (1 - frac_u) + c111 * frac_u;

                float c0 = c00 * (1 - frac_v) + c10 * frac_v;
                float c1 = c01 * (1 - frac_v) + c11 * frac_v;

                float result = c0 * (1 - frac_w) + c1 * frac_w;
                return result;
           }

           __device__ float4 ApplyTransform(const float* FluoroToCt, float4 Point)
            {
                float x = FluoroToCt[0] * Point.x + FluoroToCt[1] * Point.y + FluoroToCt[2] * Point.z + FluoroToCt[3];
                float y = FluoroToCt[4] * Point.x + FluoroToCt[5] * Point.y + FluoroToCt[6] * Point.z + FluoroToCt[7];
                float z = FluoroToCt[8] * Point.x + FluoroToCt[9] * Point.y + FluoroToCt[10] * Point.z + FluoroToCt[11];
                float w = 0.0f;
                float4 transPoint = make_float4(x,y,z,w);
                return transPoint;
            }

            __device__ bool intersectBox(float4 f4OriginMM, float4 f4DirectionMM, float4 f4MinBoundBoxMM, float4 f4MaxBoundBoxMM, 
                                        float* fMinIntersect, float* fMaxIntersect)
            {
                float4 f4Bot = make_float4((f4MinBoundBoxMM.x-f4OriginMM.x)/f4DirectionMM.x,(f4MinBoundBoxMM.y-f4OriginMM.y)/f4DirectionMM.y,
                                            (f4MinBoundBoxMM.z-f4OriginMM.z)/f4DirectionMM.z,(f4MinBoundBoxMM.w-f4OriginMM.w)/f4DirectionMM.w);
                float4 f4Top = make_float4((f4MaxBoundBoxMM.x-f4OriginMM.x)/f4DirectionMM.x,(f4MaxBoundBoxMM.y-f4OriginMM.y)/f4DirectionMM.y,
                                            (f4MaxBoundBoxMM.z-f4OriginMM.z)/f4DirectionMM.z,(f4MaxBoundBoxMM.w-f4OriginMM.w)/f4DirectionMM.w);

                float4 f4Min = make_float4(fminf(f4Bot.x, f4Top.x), fminf(f4Bot.y, f4Top.y), fminf(f4Bot.z, f4Top.z), 0.0f);
                float4 f4Max = make_float4(fmaxf(f4Bot.x, f4Top.x), fmaxf(f4Bot.y, f4Top.y), fmaxf(f4Bot.z, f4Top.z), 0.0f);

                float fLargeMin = fmaxf(fmaxf(f4Min.x, f4Min.y), f4Min.z);
                float fSmallMax = fminf(fminf(f4Max.x, f4Max.y), f4Max.z);

                *fMinIntersect = fLargeMin;
                *fMaxIntersect = fSmallMax;

                return (fSmallMax > fLargeMin);
            }

            __device__ float4 normalize(float4 v){
                float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
                if (norm > 0)
                {
                    float inv_norm = 1.0f / norm;
                    v.x *= inv_norm;
                    v.y *= inv_norm;
                    v.z *= inv_norm;
                }
                else
                {
                    v.x = 0.0f;
                    v.y = 0.0f;
                    v.z = 0.0f;
                }
                return v;
            }


            __global__ void trilinear_interpolation1(float* data, int dim_z, int dim_y, int dim_x,
                                    float CTdeltax, float CTdeltay, float CTdeltaz,
                                    float* result,int DRR_height,int DRR_width,
                                    float *FluoroToCT,float4 *ImgRay) {

                int i = blockIdx.x * blockDim.x + threadIdx.x;
                int j = blockIdx.y * blockDim.y + threadIdx.y;
                if (i < DRR_height && j < DRR_width){
                    int index = j * DRR_height + i;
                    float4 Ray = ImgRay[index];

                    float4 pointZ0 = make_float4(Ray.x,Ray.y,0.0f,0.0f);
                    float4 pointZ1 = make_float4(Ray.z,Ray.w,1.0f,0.0f);

                    if (pointZ1.x > 999 || pointZ1.y > 999)
                    {
                        return;
                    }


                    float4 Z0CT =  ApplyTransform(FluoroToCT,pointZ0);

                    float4 Z1CT = ApplyTransform(FluoroToCT,pointZ1);


                    float4 OriginCT = Z0CT;
                    float4 RayCT_ = make_float4(Z1CT.x-Z0CT.x,Z1CT.y-Z0CT.y,Z1CT.z-Z0CT.z,Z1CT.w-Z0CT.w);
                    float4 RayCT = normalize(RayCT_);

                    float T1 = 10.0;
                    float T2 = 1200.0;

                    float4 ImgDeltas = make_float4(CTdeltax,CTdeltay,CTdeltaz,1.0);
                    float RayVal = 0.0f;

                    float4 CTMinBox = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
                    float4 CTMaxBox = make_float4(-1*dim_x, -1*dim_y,-1*dim_z, 1.0f);
                    CTMaxBox.x = CTMaxBox.x * ImgDeltas.x;
                    CTMaxBox.y = CTMaxBox.y * ImgDeltas.y;
                    CTMaxBox.z = CTMaxBox.z * ImgDeltas.z;
                    CTMaxBox.w = CTMaxBox.w * ImgDeltas.w;

                    bool bRun = intersectBox(OriginCT,RayCT,CTMinBox,CTMaxBox,&T1,&T2);

                    float ValidDist = 0.0f;

                    if (bRun){
                        T1 = max(T1,0.0);
                        for (float T=T1;T<=T2;T=T+0.6){
                            float4 Loc;
                            Loc.x = -1*(OriginCT.x+T*RayCT.x)/ ImgDeltas.x;
                            Loc.y = -1*(OriginCT.y+T*RayCT.y)/ ImgDeltas.y;
                            Loc.z = -1*(OriginCT.z+T*RayCT.z)/ ImgDeltas.z;
                            Loc.w = 0.0f;

                            float LocVal = trilinear_interpolation(Loc.x,Loc.y,Loc.z,data,dim_z, dim_y, dim_x);

                            if (LocVal > -1400.0f)
                            {
                                ValidDist += 1;
                            }
                            if(LocVal<-1000.0){
                                LocVal = -1000.0;
                            }
                            if(LocVal>2000.0){
                                LocVal = 2000.0;
                            }
                            if (LocVal >= 600.0f)
                            {
                                LocVal *= 6.0f;
                            }
                            else if (LocVal < 600.0f && LocVal > 50.0f)
                            {
                                LocVal *= 3.0f;
                            }
                            else
                            {
                                LocVal *= 1.0f;
                            }

                            float Val = LocVal / 1000.0f + 1.0f;
                            RayVal = RayVal + Val;
                        }

                    }

                    if (ValidDist <= 0.0001f)
                    {
                        result[index] = 0.0;
                    }
                    else
                    {   
                        result[index] = expf(-RayVal / 200.0f);
                    }
                }
           }
        """)


    def NormalizaDRR(self, DRR):
        # 在CUDA设备上执行归一化操作
        min_val = torch.min(DRR)
        max_val = torch.max(DRR)
        normalized_data = (DRR - min_val) / (max_val - min_val)
        return normalized_data

    def GetDRR(self, path, pose):
        # ImgRay = self.getRay.GetImgRay()

        ct_spacing = self.ct_spacing
        ct_dim = self.ct_dim


        width = ct_dim[0]
        height = ct_dim[1]
        depth = ct_dim[2]

        pose = np.array(pose)  # 将元组转换为NumPy数组
        pose[-3] -= width/2
        pose[-2] -= self.src_det
        pose[-1] -= depth/4

        print("pose: ", pose)

        Pose_carmToct = CarmToCT_Pose(pose)
        trans_mat = Pose_carmToct.transform_mat()
        t = np.zeros([1, 4], dtype=np.float32)
        trans_mat = np.concatenate((trans_mat, t), axis=0)
        trans_mat = trans_mat.reshape(-1).astype(np.float32)
        # print("rot_mat: ", trans_mat)
        trans_mat_gpu = cuda.mem_alloc(trans_mat.nbytes)
        cuda.memcpy_htod(trans_mat_gpu, trans_mat)


        linear_interpolation = self.mod.get_function("trilinear_interpolation1")

        block_dim = (32, 32, 1)
        grid_dim = (
            (self.nSyVid + block_dim[0] - 1) // block_dim[0],
            (self.nSxVid + block_dim[1] - 1) // block_dim[1],
            1
        )

        linear_interpolation(
            self.d_data,
            np.int32(depth),
            np.int32(height),
            np.int32(width),
            np.float32(ct_spacing[0]),
            np.float32(ct_spacing[1]),
            np.float32(ct_spacing[2]),
            self.DRR,
            np.int32(self.nSyVid),
            np.int32(self.nSxVid),
            trans_mat_gpu,
            self.ImgRay_gpu,
            block=block_dim,
            grid=grid_dim
        )

        trans_mat_gpu.free()

        rotated_drr = self.DRR.T.flip(0)
        # print(rotated_drr)
        drr_norm = self.NormalizaDRR(rotated_drr)

        drr_numpy = drr_norm.cpu().numpy()

        # expanded_drr_tensor = drr_norm.unsqueeze(0).unsqueeze(0)
        return drr_numpy
