import logging
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.morphology import skeletonize
from skimage.util import invert

logging.basicConfig(level=logging.WARNING)
sys.setrecursionlimit(1000000)
def track_back():
    pass
def cal_arroud(n,img):
    shape = img.shape
    y = n[0]
    x = n[1]
    ls = []
    for n in [1,0,-1]:
        for m in [1,0,-1]:
            xnew = x+m
            ynew = y+n
            if xnew < shape[1] and xnew >= 0 and ynew < shape[0] and ynew >= 0:  
                l = [ynew,xnew]
                ls.append(l)
            else:
                continue
    logging.info(ls)
    return ls
def findones(ls,img):
    onelist = []
    for n,item in enumerate(ls):#迭代周围的位坐标，找到==1的点的位坐标
        isone = img[item[0],item[1]]
        if isone == 1:
            onelist.append(item)#将这个点保存在one list，用于后续根据这个点继续找相邻像素
    logging.info('该点周围存在像素的位置：{0}'.format(onelist))
    return(onelist)        

def do_point(point,img,newline):
    logging.info('选择的点：{0}'.format(point))
    ls = cal_arroud(point,img)#根据这个像素点找周围存在的像素点
    onelist = findones(ls,img)
    if len(onelist) >= 1:
        for n,item in enumerate(onelist):
            img[item[0],item[1]] = 0 #在位图中删除这些像素点
            newline.append(item) #将这些点增加到新矢量图队列
        for n,item in enumerate(onelist):    
            do_point(item,img,newline)
        return (img,newline )   
    if len(onelist) == 0:

        return (img,newline )    
#建立一副位图

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
img = cv2.imread('realpcb1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img_gray,70,255,cv2.THRESH_BINARY)#二值化
tmp = np.hstack((img_gray, thresh))  # 两张图片横向合并（便于对比显示）
# cv2.imshow('do', tmp)
# cv2.waitKey(0)



# Invert the horse image
# image = invert(data.horse())
thresh[thresh == 255] = 1
# perform skeletonization
skeleton = skeletonize(thresh)

# display results
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
#                          sharex=True, sharey=True)

# ax = axes.ravel()

# ax[0].imshow(thresh, cmap=plt.cm.gray)
# ax[0].axis('off')
# ax[0].set_title('original', fontsize=20)

# ax[1].imshow(skeleton, cmap=plt.cm.gray)
# ax[1].axis('off')
# ax[1].set_title('skeleton', fontsize=20)

# fig.tight_layout()
# plt.show()
# fig, (ax1, ax2) = plt.subplots(ncols = 2, sharex = True, sharey = True)


# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构



plt.imshow(skeleton, cmap="gray")
plt.show()
shape = skeleton.shape
logging.info(shape)
index = np.argwhere(skeleton == 1)
newlines = []

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,
                         sharex=True, sharey=True)
ax2.axis('image')
ax1.imshow(skeleton, cmap="gray")

while len(index) != 0:
    skeleton[index[0][0],index[0][1]] = 0 #在位图中删除这个像素点
    newline = []
    newline.append([index[0][0],index[0][1]]) #将该点增加到新矢量图队列
    skeleton,newline = do_point(index[0],skeleton,newline) #递归找点

    # newline = zuobiaopaixu(newline)
    # print(newline)
    newline = sorted(newline)
    newline = np.array(newline) 

    xdata = newline[:,1]
    ydata = newline[:,0] 
    ax2.plot(xdata, ydata,linewidth = 1)
    plt.draw()  
    # plt.pause(1e-27)
    newlines.append(newline)      
    index = np.argwhere(skeleton == 1)






# for newline in newlines:
#     xdata = newline[:,1]
#     ydata = newline[:,0] 
#     ax2.plot(xdata, ydata, 's',linewidth = 1)
#     plt.draw()
#     logging.info('newline:{0}'.format(newline))
plt.show()    
