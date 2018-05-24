# -*- coding: UTF-8 -*-
from win32com.client import Dispatch, VARIANT
import pythoncom
import os
import matplotlib.pyplot as plt
import numpy as np

a = Dispatch('AutoCAD.Application')
print(a)
a.Visible = 1

import cv2
import matplotlib.pyplot as plt
def track_back(x):
    pass

def POINT(x,y,z):
   return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (x,y,z))  

path=os.path.dirname(__file__)

img = cv2.imread(path+'/2.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True,
                       figsize=(8, 4))
# 自适应均衡化，参数可选
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl1 = clahe.apply(img_gray)
# cv2.imshow('equalization', np.hstack((img_gray, cl1)))  # 并排显示
# cv2.waitKey(0)

# cv2.namedWindow('br/co')
# cv2.createTrackbar('alpha', 'br/co', 100, 200, track_back)
# cv2.createTrackbar('beta', 'br/co', 0, 255, track_back)
# while(True):
#     # 获取滑动条的值
#     alpha_val = cv2.getTrackbarPos('alpha', 'br/co')
#     beta_val = cv2.getTrackbarPos('beta', 'br/co')

#     res = np.uint8(np.clip((alpha_val/100 * img_gray + beta_val), 0, 255))
#     tmp = np.hstack((img_gray, res))  # 两张图片横向合并（便于对比显示）
#     cv2.imshow('br/co', tmp)

#     # 按下ESC键退出
#     if cv2.waitKey(30) == 27:
#         break


# cv2.namedWindow('filt')
# # img = cv2.medianBlur(img, 5)  # 中值滤波
# # img = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
# # img = cv2.bilateralFilter(img_gray, 9, 75, 75)  # 双边滤波
# cv2.createTrackbar('d', 'filt', 9, 16, track_back)
# cv2.createTrackbar('sigma', 'filt', 0, 255, track_back)

# while(True):
#     # 获取滑动条的值
#     d_val = cv2.getTrackbarPos('d', 'filt')
#     sigma_val = cv2.getTrackbarPos('sigma', 'filt')
#     img_filt = cv2.bilateralFilter(res, d_val, sigma_val, sigma_val)  # 双边滤波
#     tmp = np.hstack((img_gray, img_filt))  # 两张图片横向合并（便于对比显示）
#     cv2.imshow('filt', tmp)

#     # 按下ESC键退出
#     if cv2.waitKey(30) == 27:
        # break

# 二值化图像
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# ret_deal, thresh_deal = cv2.threshold(img_filt, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# tmp = np.hstack((thresh, thresh_deal))  # 两张图片横向合并（便于对比显示）
ax.imshow(thresh)
plt.show()
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)
# 寻找二值化图中的轮廓
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# image_deal, contours_deal, hierarchy_deal = cv2.findContours(thresh_deal, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

# print(len(contours))  # 结果应该为2c

img_z1 = np.zeros((img.shape[0],img.shape[1],3), np.uint8)     
# img_z2 = np.zeros((img.shape[0],img.shape[1],3), np.uint8)  
img_contours = cv2.drawContours(img_z1, contours, -1, (255,255,255), 1)
# img_contours_deal = cv2.drawContours(img_z2, contours_deal, -1, (255,255,255), 1)
cv2.imshow('img_contours',img_contours)
cv2.waitKey(0)

# fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
#                        figsize=(8, 4))
# ax[0].imshow(img_contours)
# ax[0].set_axis_off()
# ax[1].imshow(img_contours_deal)
# ax[1].set_axis_off()
# plt.show()
# img_gray_deal = cv2.cvtColor(img_contours_deal, cv2.COLOR_BGR2GRAY)

# 3.霍夫圆变换
# drawing = np.zeros(img.shape[:], dtype=np.uint8)

# circles = cv2.HoughCircles(img_gray_deal, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=65)
# circles = np.int0(np.around(circles))

# # 将检测的圆画出来
# for i in circles[0, :]:
#     cv2.circle(img_contours_deal, (i[0], i[1]), i[2], (0, 255, 0), 1)  # 画出外圆
#     cv2.circle(img_contours_deal, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画出圆心

# cv2.imshow('circles', np.hstack((img_contours_deal, drawing)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

count_all = 0
for cnt in contours:
    count_all = count_all + len(cnt)

count_cnt = 0
for cnt in contours:
    if len(cnt) < 3:
        continue

    t = None
    for point in cnt:
        count_cnt = count_cnt + 1
            #计算完成百分比

        xdata = point[0,0]
        ydata = point[0,1]
        start_pt = POINT(xdata,ydata,0)
        if t == None:
            t = start_pt
            first_point = start_pt
            continue
        else:
            end_pt = t
            t = start_pt

        # a.ActiveDocument.ModelSpace.AddLine(start_pt,end_pt)
    complie_cnt = int(count_cnt/count_all*100)
    print('complied {0}%'.format(complie_cnt),end = '\r',flush = True)
    # a.ActiveDocument.ModelSpace.AddLine(first_point,start_pt)
# a.ZoomAll() 
#test1
