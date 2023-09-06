# -*- coding: UTF-8 -*-
'''
@Project ：GeneratorLabelData 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：soldier Hou
@E-mail  : 17853538105@163.com
@Date    ：2023/8/31 15:59 
'''
import os
import cv2

path = r"D:\PythonCode\PatientRegistration\PatientRegistration\datasets\220303\220303.png"
drr_numpy = cv2.imread(path)
cv2.imshow("drr Image", drr_numpy)
cv2.waitKey(0)
cv2.destroyAllWindows()

save_path = "../data"
pose = [0, 0, 0, 0, 0, 0]
pose_str = '_'.join(map(str, pose))
folder_path = os.path.join(save_path,pose_str)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    cv2.imwrite(folder_path+"/DRR.png",drr_numpy)
    print(f"Folder '{folder_path}' created successfully.")