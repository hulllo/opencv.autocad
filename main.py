# -*- coding: UTF-8 -*-
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pythoncom
from matplotlib.widgets import RadioButtons, Slider
from skimage import data
from skimage.morphology import skeletonize
from skimage.util import invert
from win32com.client import VARIANT, Dispatch

# a = Dispatch('AutoCAD.Application')
# print(a)
# a.Visible = 1
def pro_contour(contour):
    xdata = contour[:,:,0]
    ydata = contour[:,:,1]
    # print(xdata)


    start = np.where(xdata == np.min(xdata))[0][-1] #找出开始点位置，x轴的最小值
    stop = np.where(xdata == np.max(xdata))[0][0]#找出结束点位置，x轴的最大值
    x = xdata[start:stop,0] #从开始点到结束点的x轴值
    y = ydata[start:stop,0]#从开始点到结束点的y轴值

    x,ynews = stretch_line(x,y) #找到可能的直线拉直
    # print(x)
    # plot1 = plt.plot(x, y, 's' ,label='original values')  
    # plot2 = plt.plot(x, ynews,label='original values')


    ############################################################################
    start1 = start - 1  #另一端曲线的‘开始’
    stop1 = stop + 1##另一端曲线的'结束'
    xdata1 = xdata[0:start1,0]  #从开始点到结束点的x轴值
    xdata1 = xdata1[::-1]
    xdata2 = xdata[stop1:-1,0]
    xdata2 = xdata2[::-1]
    x = np.hstack((xdata1, xdata2))
    ydata1 = ydata[0:start1,0]  #从开始点到结束点的y轴值
    ydata1 = ydata1[::-1]

    ydata2 = ydata[stop1:-1,0]
    ydata2 = ydata2[::-1]

    y = np.hstack((ydata1, ydata2))
    x,ynews = stretch_line(x,y) #找到可能的直线拉直
    # plot1 = plt.plot(x, y,'s',label='original values')  
    plot2 = plt.plot(x, ynews,label='original values')

def find_max_index(lists):
    listsnew = lists.copy()
    listsnew = set(listsnew)
    listsnew = sorted(listsnew, reverse = True)
    for item in listsnew:
        index = lists.index(item)
        yield item,index
def track_back(x):
    pass

def POINT(x,y,z):
   return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (x,y,z))  

def stretch_line(x,y):
    ynews = y.copy()
    deltas = []
    #获得x轴点两两之间的距离
    for n in range(len(x)-1):
        delta = x[n+1] - x[n]
        deltas.append(delta)  

    for n, x_item in enumerate(x):
        for item,index in find_max_index(deltas):
            if item < 6:
                break
            k = (y[index+1] - y[index])/(x[index+1]-x[index])  #计算直线斜率
            d = y[index] - k * x[index]  #计算直线截距
            try:
                y_delta = abs(k * x[n] + d - y[n])
                y_delta_pre = abs(k * x[n-1] + d - y[n-1])
                y_delta_next = abs(k * x[n+1] + d - y[n+1])
            except IndexError:
                pass
            try:
                if y_delta <= 3 and y_delta_pre <=2 and y_delta_next <=2:
                    ynews[n] = k * x[n] + d #y值相差小，替换
                    # print('替换success')
            except IndexError:
                pass
    
    # print('x:', len(x))
    # print('ynews:', len(ynews))
    return x, ynews
def main():
    path=os.path.dirname(__file__)

    img = cv2.imread(path+'/realpcb1.jpg')

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        

    cv2.namedWindow('do',0)
    cv2.createTrackbar('alpha', 'do', 100, 200, track_back)
    cv2.createTrackbar('beta', 'do', 0, 255, track_back)
    cv2.createTrackbar('d', 'do', 9, 16, track_back)
    cv2.createTrackbar('sigma', 'do', 0, 255, track_back)
    cv2.createTrackbar('thresh', 'do', 70, 255, track_back)
    cv2.createTrackbar('median', 'do', 3, 10, track_back)

    while(True):
        # 获取滑动条的值
        alpha_val = cv2.getTrackbarPos('alpha', 'do')
        beta_val = cv2.getTrackbarPos('beta', 'do')
        d_val = cv2.getTrackbarPos('d', 'do')
        sigma_val = cv2.getTrackbarPos('sigma', 'do')
        thresh_val = cv2.getTrackbarPos('thresh', 'do')
        median_val = cv2.getTrackbarPos('median', 'do')

        res = np.uint8(np.clip((alpha_val/100 * img_gray + beta_val), 0, 255))#亮度，对比度
        img_filt = cv2.bilateralFilter(res, d_val, sigma_val, sigma_val)  # 双边滤波
        ret,thresh = cv2.threshold(img_filt,thresh_val,255,cv2.THRESH_BINARY)#二值化
        thresh = cv2.medianBlur(thresh, median_val*2-1)  # 中值滤波

        tmp = np.hstack((img_gray, thresh))  # 两张图片横向合并（便于对比显示）
        cv2.imshow('do', tmp)

        # 按下ESC键退出
        if cv2.waitKey(30) == 27:
            break


    # Invert the horse image
    # image = invert(data.horse())
    thresh[thresh == 255] = 1
    # perform skeletonization
    skeleton = skeletonize(thresh)

    # display results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                            sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(thresh, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()
    fig, (ax1, ax2) = plt.subplots(ncols = 2, sharex = True, sharey = True)


    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构



    ax1.imshow(thresh, cmap="gray")

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)
    # 寻找二值化图中的轮廓
    # image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # image_deal, contours_deal, hierarchy_deal = cv2.findContours(thresh_deal, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)
    # image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

    ax2.axis('image')
    ##################################################################


    for n, contour in enumerate(contours):
        pro_contour(contour)
    plt.show()
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

if __name__ == '__main__':
    main()