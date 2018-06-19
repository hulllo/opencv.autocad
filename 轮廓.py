# -*- coding: UTF-8 -*-
from win32com.client import Dispatch, VARIANT
import pythoncom
import os
import numpy as np
# a = Dispatch('AutoCAD.Application')
# a.Visible = 1
from skeleton_to_vertor1_1 import del_Duplicate_pt
from skeleton_to_vertor1_1 import stretch_line
import cv2
import matplotlib.pyplot as plt
from count_same import count_same


def POINT(x,y,z):
   return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (x,y,z))  

path=os.path.dirname(__file__)

img = cv2.imread(path+'/realpcb1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.medianBlur(img_gray, 5) #中值滤波
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# 寻找二值化图中的轮廓
image, contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(len(contours))  # 结果应该为2c
img = cv2.drawContours(img, contours, -1, (255,255,0), 1)

for i, v in enumerate(contours):
    x_datas = v[:,0,0]
    y_datas = v[:,0,1]

    x_datas, y_datas = del_Duplicate_pt(x_datas,y_datas)#删除同一直线上冗余的坐标
    x_datas, y_datas = stretch_line(x_datas, y_datas)#拉直直线
    x_datas, y_datas = del_Duplicate_pt(x_datas,y_datas)#删除同一直线上冗余的坐标   
    plt.plot(x_datas, y_datas,'s')
    plt.plot(x_datas, y_datas)
    plt.draw()
    plt.pause(1e-27)
plt.show()



# for cnt in contours:
#     t = None
#     for point in cnt:
#         xdata = point[0,0]
#         ydata = point[0,1]
#         start_pt = POINT(xdata,ydata,0)
#         if t == None:
#             t = start_pt
#             first_point = start_pt
#             continue
#         else:
#             end_pt = t
#             t = start_pt

#         # a.ActiveDocument.ModelSpace.AddLine(start_pt,end_pt)
#     start_pt = end_pt    
#     end_pt = first_point
#     # a.ActiveDocument.ModelSpace.AddLine(start_pt,end_pt)