import logging
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pythoncom
from skimage import data
from skimage.morphology import skeletonize
from skimage.util import invert
from win32com.client import VARIANT, Dispatch
from count_same import count_same

logging.basicConfig(level=logging.WARNING)
sys.setrecursionlimit(1000000)

def stretch_line(x, y):
    '''x, y 为narray'''
    

def track_back():
    pass
def cal_arroud(n,img):
    shape = img.shape
    y = n[0]
    x = n[1]
    arroud_point = []
    for n in [1,0,-1]:
        for m in [1,0,-1]:
            xnew = x+m
            ynew = y+n
            if xnew < shape[1] and xnew >= 0 and ynew < shape[0] and ynew >= 0:  
                l = [ynew,xnew]
                arroud_point.append(l)
            else:
                continue
    logging.info(arroud_point)
    return arroud_point
def findones(arroud_point,img):
    onelist = []
    for n,item in enumerate(arroud_point):#迭代周围的位坐标，找到==1的点的位坐标
        isone = img[item[0],item[1]]
        if isone == 1:
            onelist.append(item)#将这个点保存在one list，用于后续根据这个点继续找相邻像素
    logging.info('该点周围存在像素的位置：{0}'.format(onelist))
    return(onelist)        

def do_point(point,img,newline,newlines = []):
    # print('选择的点：{0}'.format(point))
    arroud_point = cal_arroud(point,img)#根据这个像素点找周围存在的像素点
    onelist = findones(arroud_point,img)

    if len(onelist) >= 1:
        for n,item in enumerate(onelist):
            img[item[0],item[1]] = 0 #在位图中删除这些像素点
            if n >= 1:
                newline.append([point[0],point[1]])
            newline.append(item) #将这些点增加到新矢量图队列
            img, newlines = do_point(item,img,newline,newlines)
            if newline in newlines or len(newline) <= 10:
                newline = []
                continue
            else:
                newlines.append(newline)
            # print('有{0}条线：'.format(len(newlines)))
            newline = []
        return (img,newlines )   

    if len(onelist) == 0:
        return (img,newlines )    

def zuobiaopaixu(a):  
    b=[]  
    l=len(a)  
    for i in range(l):  
        j=i  
        for j in range(l):  
            if (a[i][0]<a[j][0]):  
                a[i],a[j]=a[j],a[i]  
            if (a[i][1]>a[j][1]):  
                a[i],a[j]=a[j],a[i]  
  
    for k in range(len(a)):  
        b.append(a[k])  
    return b  

def open_(filename):
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.medianBlur(img_gray, 5) #中值滤波

    # 固定阈值
    # ret,thresh = cv2.threshold(img_gray,70,255,cv2.THRESH_BINARY)#二值化

    # 自适应阈值    
    # thresh = cv2.adaptiveThreshold(
    # img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)

    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(thresh)

    # tmp = np.hstack((img_gray, thresh))  # 两张图片横向合并（便于对比显示）
    thresh[thresh == 255] = 1
    # perform skeletonization
    skeleton = skeletonize(thresh)
    # plt.imshow(skeleton, cmap="gray")
    # plt.show()
    return skeleton

def POINT(x,y,z):
   return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (x,y,z))  

img = np.array([[1,1,1,1,1,1],
            [0,1,0,0,1,0],
            [0,1,1,0,0,0],
            [1,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,0] ])

# img = np.array([[0,0,0,0,0,0],
#             [0,0,0,0,0,0],
#             [0,0,0,0,0,0],
#             [0,0,0,0,0,0],
#             [0,0,0,0,0,0],
#             [0,0,0,0,0,0] ])
img = open_('minirealpcb1.jpg')
shape = img.shape
logging.info(shape)
index = np.argwhere(img == 1)
newlines = []

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,
                         sharex=True, sharey=True)

lines = []
# a = Dispatch('AutoCAD.Application')
# print(a)
# a.Visible = 1
ax2.axis('image')
ax1.imshow(img, cmap="gray")
while len(index) != 0:
    img[index[0][0],index[0][1]] = 0 #在位图中删除这个像素点
    newline = []
    newline.append([index[0][0],index[0][1]]) #将该点增加到新矢量图队列
    img,newlines = do_point(index[0],img,newline,newlines = []) #递归找点

    for x in newlines:
        for n, point in enumerate(x):
            # print(point)
            start_point = POINT(x[n][1], shape[0]-x[n][0], 0)
            try:
                stop_point = POINT(x[n+1][1], shape[0]-x[n+1][0], 0)
            except IndexError :
                continue
            # a.ActiveDocument.ModelSpace.AddLine(start_point, stop_point)
        x = np.array(x)
        xdatas = x[:,1]
        ydatas = x[:,0]
        ax2.plot(xdatas, ydatas,linewidth = 1)
       
        plt.draw()
        # plt.pause(1e-27)
        ks = []
        for n,xdata in enumerate(xdatas):
            if n >= 1:
                k = (ydatas[n]-ydatas[n-1])/(xdatas[n]-xdatas[n-1])
                ks.append(k)
        same_v,same_count = count_same(ks)
        print(same_v,'\n',same_count)
        count = 0
        for i,v in enumerate(same_count):
            if v == 1:
                count =count + 1
            else:
                start_c = i - count #重复发生之前的索引
                stop_c = i #重复结束后的索引
                # print('start_c:',start_c)
                # print('stop_c:',stop_c)
                count = 0
                if same_count[start_c-1]>4 and same_count[stop_c] > 4:

                    real_start_index = 0
                    for ii,vv in enumerate(same_count[0:start_c]):
                        real_start_index = real_start_index + vv
                    print('real_start_index:',real_start_index)
                    
                    real_stop_index = 0
                    for ii,vv in enumerate(same_count[0:stop_c]):
                        real_stop_index = real_stop_index + vv
                    print('real_stop_index:',real_stop_index)

                    if xdatas[real_start_index]==xdatas[real_stop_index+1] or ydatas[real_start_index]==ydatas[real_stop_index+1]:
                        newxdatas = np.delete(xdatas, list(range(real_start_index+1,real_stop_index)))
                        newydatas = np.delete(ydatas, list(range(real_start_index+1,real_stop_index)))

    

                else:
                    continue

        ax2.plot(newxdatas, newydatas,linewidth = 1)

        # break
    # break
    index = np.argwhere(img == 1)
plt.show()    





# for line in lines:
#     for n, point in enumerate(line):
#         print(point)
#         start_point = POINT(line[n][1], line[0][0], 0)
#         try:
#             stop_point = POINT(line[n+1][1], line[n+1][0], 0)
#         except IndexError :
#             continue
#         a.ActiveDocument.ModelSpace.AddLine(start_point, stop_point)
# a.ZoomAll() 
