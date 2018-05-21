# -*- coding: UTF-8 -*-
from win32com.client import Dispatch, VARIANT
import pythoncom
import os

a = Dispatch('AutoCAD.Application')
a.Visible = 1

import cv2
import matplotlib.pyplot as plt


def POINT(x,y,z):
   return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (x,y,z))  

path=os.path.dirname(__file__)

img = cv2.imread(path+'/3.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# 寻找二值化图中的轮廓
image, contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(len(contours))  # 结果应该为2c
# img = cv2.drawContours(img, contours, 1, (255,255,0), 2)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cnt = contours[5]
# epsilon = 0.01*cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,epsilon,True)
# img = cv2.drawContours(img, [approx], 0, (255,255,0), 2)
# cv2.imshow('img',img)

# cv2.waitKey(0)
for cnt in contours:
    t = None
    for point in cnt:
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

        a.ActiveDocument.ModelSpace.AddLine(start_pt,end_pt)
    start_pt = end_pt    
    end_pt = first_point
    a.ActiveDocument.ModelSpace.AddLine(start_pt,end_pt)
