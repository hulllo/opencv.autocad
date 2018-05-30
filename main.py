# -*- coding: UTF-8 -*-
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pythoncom
from matplotlib.widgets import RadioButtons, Slider
from win32com.client import VARIANT, Dispatch

# a = Dispatch('AutoCAD.Application')
# print(a)
# a.Visible = 1
def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def track_back(x):
    pass

def POINT(x,y,z):
   return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (x,y,z))  

path=os.path.dirname(__file__)

img = cv2.imread(path+'/realpcb1.jpg')

# fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True,
#                        figsize=(8, 4))
# b = img[:, :, 0]
# g = img[:, :, 1]
# r = img[:, :, 2]
# ax[0].imshow(b, cmap="gray")
# ax[1].imshow(g, cmap="gray")
# ax[2].imshow(r, cmap="gray")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


                      
# 自适应均衡化，参数可选
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl1 = clahe.apply(img_gray)
# cv2.imshow('equalization', np.hstack((img_gray, cl1)))  # 并排显示
# cv2.waitKey(0)

# cv2.namedWindow('br/co',0)
# cv2.createTrackbar('alpha', 'br/co', 100, 200, track_back)
# cv2.createTrackbar('beta', 'br/co', 0, 255, track_back)
# while(True):
#     # 获取滑动条的值
#     alpha_val = cv2.getTrackbarPos('alpha', 'br/co')
#     beta_val = cv2.getTrackbarPos('beta', 'br/co')

#     res = np.uint8(np.clip((alpha_val/100 * r + beta_val), 0, 255))
#     tmp = np.hstack((img_gray, res))  # 两张图片横向合并（便于对比显示）
#     cv2.imshow('br/co', tmp)

#     # 按下ESC键退出
#     if cv2.waitKey(30) == 27:
#         break


# cv2.namedWindow('filt')
# # img = cv2.medianBlur(img, 5)  # 中值滤波
# # img = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
# img_gray = cv2.bilateralFilter(img_gray, 9, 75, 75)  # 双边滤波
# cv2.createTrackbar('d', 'filt', 9, 16, track_back)
# cv2.createTrackbar('sigma', 'filt', 0, 255, track_back)

# while(True):
#     # 获取滑动条的值
#     d_val = cv2.getTrackbarPos('d', 'filt')
#     sigma_val = cv2.getTrackbarPos('sigma', 'filt')
#     img_filt = cv2.bilateralFilter(img_gray, d_val, sigma_val, sigma_val)  # 双边滤波
#     tmp = np.hstack((img_gray, img_filt))  # 两张图片横向合并（便于对比显示）
#     cv2.imshow('filt', tmp)

    # 按下ESC键退出
    # if cv2.waitKey(30) == 27:
    #     break
fig, (ax1, ax2) = plt.subplots(ncols = 2, sharex = True, sharey = True)
# fig.gca().invert_yaxis()

# fig, bx = plt.subplots(ncols=2, sharex=True, sharey=True,)                       
# 二值化图像
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret_deal, thresh_deal = cv2.threshold(img_filt, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# tmp = np.hstack((thresh, thresh_deal))  # 两张图片横向合并（便于对比显示）
# cv2.namedWindow('thresh', 0)
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)


# 2.定义结构元素
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
# print(kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
# print(kernel)

# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构
# print(kernel)

# 闭运算
thresh = cv2.medianBlur(thresh, 5)  # 中值滤波

closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 开运算
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# # 5.顶帽
# tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)

# # 6.黑帽
# blackhat = cv2.morphologyEx(thresh, cv2.MORPH_BLACKHAT, kernel)

# cv2.imshow('closing', np.hstack((thresh, closing)))
# bx[0].imshow(thresh, cmap="gray")
# bx[1].imshow(closing, cmap="gray")

# cv2.waitKey(0)


ax1.imshow(thresh, cmap="gray")

image, contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)
# 寻找二值化图中的轮廓
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# image_deal, contours_deal, hierarchy_deal = cv2.findContours(thresh_deal, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

# print(len(contours))  # 结果应该为2c

# img_z1 = np.zeros((img.shape[0],img.shape[1],3), np.uint8)     
# img_z2 = np.zeros((img.shape[0],img.shape[1],3), np.uint8)  
# img_contours = cv2.drawContours(img_z1, contours, -1, (255,255,255), 1)
# img_contours_deal = cv2.drawContours(img_z2, contours_deal, -1, (255,255,255), 1)
# ax[1].imshow(img_contours, cmap="gray")
ax2.axis('image')
# ax2.set_xticks([])#关闭坐标轴
# ax2.set_yticks([])

##################################################################
n =61
xdata = contours[n][:,:,0]
ydata = contours[n][:,:,1]
# print(xdata.shape)

start = np.where(xdata == np.min(xdata))[0][-1] #找出开始点位置，x轴的最小值
stop = np.where(xdata == np.max(xdata))[0][0]#找出结束点位置，x轴的最大值
# print(start)
# print(stop)
x = xdata[start:stop,0] #从开始点到结束点的x轴值
y = ydata[start:stop,0]#从开始点到结束点的y轴值
# print(xdata[start:stop,0])
deltas = []
for n in range(len(x)-1):
    delta = x[n+1] - x[n]
    deltas.append(delta)
print(deltas)
max_index = deltas.index(max(deltas))
print (max_index) # 返回最大值
print(x[max_index],y[max_index])
print(x[max_index+1],y[max_index+1])

k = (y[max_index+1] - y[max_index])/(x[max_index+1]-x[max_index])  #计算直线斜率
d = y[max_index] - k * x[max_index]  #计算直线截距
print('y = {0}*x + ({1})'.format(k,d))

ynews = []
for n in x:
    ynew = k * n + d
    ynews.append(ynew)
print('ynews',ynews)
k1s = []
k2s = []
print('x is :',x)


deltas = abs(ynews - y)

# for n in range(len(deltas)):
#     if deltas[n] != 0:

for n, delta in enumerate(deltas):
    try:
        if delta > 2 :
            # print(delta,end = '\t')
            if deltas[n-1] > 2 or deltas[n+1] > 2:
                # print(deltas[n-1])
                ynews[n] = y[n]
        elif deltas[n-1] > 2 or deltas[n+1] > 2:
            ynews[n] = y[n]



    except IndexError:
        pass
print(deltas)
print(abs(ynews - y))

# print('k1 is :', k1s)
# print('deltal of k1 & k2 is:',np.array(k1s) - np.array(k2s))
print()
plot1 = plt.plot(x, y, 's' ,label='original values')  
plot2 = plt.plot(x, ynews,label='original values')

# #对曲线进行拟合
# f1 = np.polyfit(x, y, 2)  
# p1 = np.poly1d(f1)  
# # print(p1) 
# #也可使用yvals=np.polyval(f1, x)  
# yvals = p1(x)  #拟合y值  
# plot1 = plt.plot(x, y, 's',label='original values')  
# deltas = abs(yvals - y)#计算拟合值与实际值的差值，如果差值>4,舍弃这个拟合值。
# for n, delta in enumerate(deltas):
#     if delta > 4:
#         yvals[n] = y[n]
# ax2.plot(x, yvals, linewidth=1)
############################################################################
start1 = start - 1  #另一端曲线的‘开始’
stop1 = stop + 1##另一端曲线的'结束'
xdata1 = xdata[0:start1,0]  #从开始点到结束点的x轴值
xdata2 = xdata[stop1:-1,0]
xdata2 = xdata2[::-1]
x = np.hstack((xdata1, xdata2))
ydata1 = ydata[0:start1,0]  #从开始点到结束点的x轴值
ydata2 = ydata[stop1:-1,0]
ydata2 = ydata2[::-1]

y = np.hstack((ydata1, ydata2))

# y = ydata[start:stop,0]#从开始点到结束点的y轴值
# print(x)
# print(y)
#对曲线进行拟合
f1 = np.polyfit(x, y, 1)  
p1 = np.poly1d(f1)  
# print(p1) 
#也可使用yvals=np.polyval(f1, x)  
yvals = p1(x)  #拟合y值  
plot1 = plt.plot(x, y)  
deltas = abs(yvals - y)#计算拟合值与实际值的差值，如果差值>4,舍弃这个拟合值。
for n, delta in enumerate(deltas):
    if delta > 4:
        yvals[n] = y[n]

# ax2.plot(x, yvals, linewidth=1)


# hull = cv2.convexHull(contours[60])
# image = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
# cv2.polylines(image, [hull], True, (0, 255, 0), 2)
# cv2.imshow('convex hull', image)
# cv2.waitKey(0)




# for n, contour in enumerate(contours):
#     xdata = contour[:,:,0]
#     ydata = contour[:,:,1]
#     ax2.plot(xdata, ydata, linewidth=1)

plt.show()

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
