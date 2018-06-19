import logging
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pythoncom
from skimage import data
from skimage.morphology import skeletonize
from win32com.client import VARIANT, Dispatch
from count_same import count_same
import math

logging.basicConfig(level=logging.WARNING)
sys.setrecursionlimit(1000000000)

AUTOCAD_ENABLE = 0

def find_index(v,ls):
    '''
    查找元素在列表中的位置
    '''
    return [ i for i in range(len(ls)) if ls[i] == v]

def stretch_line(newxdatas, newydatas):
    '''
    拉直直线
    参数：
    1. newxdatas, x轴坐标组合, numpy.narray
    2. newydatas, y轴坐标组合, numpy.narray
    返回:
    1. newxdatas, x轴坐标组合, numpy.narray
    2. newydatas, y轴坐标组合, numpy.narray
    '''
    x_deltas = []
    y_deltas = []  
    ks = []  
    for i,v in enumerate(newxdatas):
        if i >= 1:
            x_deltas.append(newxdatas[i] - newxdatas[i-1])
            y_deltas.append(newydatas[i] - newydatas[i-1])
    x_deltas_sorted = sorted(x_deltas, reverse = -1) #对x两两差值进行从大到小排序
    y_deltas_sorted = sorted(y_deltas, reverse = -1) #对y两两差值进行从大到小排序
    
    #对直线进行拉直处理
    for i in range(len(x_deltas_sorted)): 
        x_deltas = []

        #刷新x两两差值的排序
        for ii,vv in enumerate(newxdatas):
            if ii >= 1:
                x_deltas.append(newxdatas[ii] - newxdatas[ii-1])
        x_deltas_sorted = sorted(x_deltas, reverse = -1)

        #x差值大于等于5，说明是一条较长的直线，计算该直线的函数
        xdelta =x_deltas_sorted[i]
        if abs(xdelta) >=5:
            x_index = find_index(xdelta, x_deltas) 
            for vvv in x_index:
                k = (newydatas[vvv+1] - newydatas[vvv])/(newxdatas[vvv+1] - newxdatas[vvv])
                d = newydatas[vvv] - newxdatas[vvv] * k

            #根据直线的函数，遍历所有的x轴坐标，实际y轴坐标与直线坐标计算的y轴坐标对比，小于1，认为是同一条直线
                for iiii,vvvv in enumerate(newxdatas):
                    try:
                        if (abs(newydatas[iiii] - k * newxdatas[iiii] - d) <= 1 
                        and abs(newydatas[iiii-1] - k * newxdatas[iiii-1] - d <= 1)
                        and abs(newydatas[iiii+1] - k * newxdatas[iiii+1] - d <= 1)):
                        
                            newydatas[iiii] = k * newxdatas[iiii] + d
                    except IndexError:
                        continue

        y_deltas = []
        #刷新y两两差值的排序
        for ii,vv in enumerate(newxdatas):
            if ii >= 1:
                y_deltas.append(newydatas[ii] - newydatas[ii-1])
        y_deltas_sorted = sorted(y_deltas, reverse = -1)
        # print('y_deltas_sorted1:',y_deltas_sorted)

        ydeltal =y_deltas_sorted[i]
        if abs(ydeltal) >= 5:
            y_index = y_deltas.index(ydeltal)   
            if newxdatas[y_index+1] - newxdatas[y_index] == 0:  #是一条垂线
                d = newxdatas[y_index]
                k = 0
                for i,v in enumerate(newydatas):
                    if abs(newxdatas[i] -  d) <= 1:
                        newxdatas[i] =  d
    return newxdatas, newydatas    

def track_back():
    pass

def cal_arroud(n,img):
    '''
    计算某个像素周围像素的坐标

    参数：
    1：n，点坐标，list
    2：img，位图数组，numpy.ndarray

    返回：
    1：arroud_point，周围的坐标点坐标，list
    '''
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
    '''
    查找位坐标列表上是1的坐标

    参数：
    1：arroud_point,位坐标列表, list
    2: img, 位图数组, numpy.narray

    返回：
    1. onelist, 值等于1的位坐标的列表, list
    '''
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
            if len(newlines) >= 3:#相连的线段太多
                return (img,newlines ) 

            if newline in newlines or len(newline) <= 10:
                newline = []
                continue
            else:
                newlines.append(newline)
                x = np.array(newline)
                xdatas = x[:,1]
                ydatas = x[:,0]
                # plt.plot(xdatas, ydatas)
                # plt.draw()
        
            # print('have {0} lines：'.format(len(newlines)))
            newline = []
        
        return (img,newlines )   
    

    if len(onelist) == 0:
        return (img,newlines )    

def open_(filename):
    '''
    打开文件为cv2对象，中值滤波，二值化，获取骨架
    参数：
    1. filename, 文件路径名字，str
    返回：
    2. skeleton, 骨架，numpy.ndarray
    '''
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.medianBlur(img_gray, 5) #中值滤波

    # 固定阈值
    # ret,thresh = cv2.threshold(img_gray,70,255,cv2.THRESH_BINARY)#二值化

    # 自适应阈值    
    # thresh = cv2.adaptiveThreshold(
    # img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)

    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # tmp = np.hstack((img_gray, thresh))  # 两张图片横向合并（便于对比显示）
    thresh[thresh == 255] = 1
    # perform skeletonization
    skeleton = skeletonize(thresh)
    # plt.imshow(skeleton, cmap="gray")
    # plt.show()
    return skeleton

def del_Duplicate_pt(xdatas,ydatas):
    ''' 删除直线上冗余的点
    参数：
        x，y轴坐标, numpy.narray
    返回：
        x，y轴坐标, numpy.narray
    '''
    ks = []
    for n,xdata in enumerate(xdatas):
        if n >= 1:
            deltalx = xdatas[n]-xdatas[n-1]
            if deltalx == 0:
                k = 'inf' 
            else:
                k = (ydatas[n]-ydatas[n-1])/deltalx #计算所有点两点之间的斜率
            ks.append(k)

    same_v,same_count = count_same(ks) #计算重复斜率的个数
    # count = 0
    del_ranges = []
    for i,v in enumerate(same_count):
        real_index = 0
        if same_count[i] >= 2:  #重复的斜率大于3，有多余的点
            # print('i:',i)
            for ii,vv in enumerate(same_count[0:i]):    #计算重复点前面的元素总个数
                real_index = real_index + vv
            real_index_st = real_index +1       #计算重复点的开始位置
            real_index_sp = real_index_st + same_count[i]-1 #计算重复点的结束位置
            del_range = list(range(real_index_st,real_index_sp))    
            del_ranges = del_ranges + del_range
    newxdatas = np.delete(xdatas, del_ranges)   #删除重复的x点
    newydatas = np.delete(ydatas, del_ranges)   #删除重复的y点
    return newxdatas, newydatas

def POINT(x,y,z):
   return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (x,y,z))  

def main():
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
    img = open_('realpcb1.jpg')
    shape = img.shape
    logging.info(shape)
    index = np.argwhere(img == 1)
    strat_index = index
    newlines = []

    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,
                            sharex=True, sharey=True)

    lines = []

    ax2.axis('image')
    ax1.imshow(img, cmap="gray")

    if AUTOCAD_ENABLE:
        a = Dispatch('AutoCAD.Application')
        a.Visible = 1
        print('连接AUTOCAD成功')

    while len(index) != 0:
        print('进度：', 100-len(index)/len(strat_index)*100, end = '\r', flush = True)
        img[index[0][0],index[0][1]] = 0 #在位图中删除这个像素点
        newline = []
        newline.append([index[0][0],index[0][1]]) #将该点增加到新矢量图队列
        img,newlines = do_point(index[0],img,newline,newlines = []) #递归找点,找到一个或连在一起的多个线段
        for x in newlines:

            x = np.array(x)
            xdatas = x[:,1]
            ydatas = x[:,0]
            newxdatas, newydatas = del_Duplicate_pt(xdatas,ydatas)#删除同一直线上冗余的坐标
            newxdatas, newydatas = stretch_line(newxdatas, newydatas)#拉直直线
            newxdatas, newydatas = del_Duplicate_pt(newxdatas,newydatas)#删除同一直线上冗余的坐标

            # ax2.plot(newxdatas, newydatas,'s',linewidth = 1)
            # ax2.plot(newxdatas, newydatas,linewidth = 1)
            # plt.draw()
            # plt.pause(1e-7)


            #对于较短距离的三个点，将中间点替换为两端点的平均值
            for i, v in enumerate(newxdatas):
                x1,y1 = newxdatas[i],newydatas[i]
                try:
                    x2,y2 = newxdatas[i+1], newydatas[i+1]
                    x3,y3 = newxdatas[i+2], newydatas[i+2]
                except IndexError:
                    continue
                length1 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
                length2 = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)  

                if length1 <= 5 and length2 <= 5:
                    newxdatas[i+1], newydatas[i+1] = (newxdatas[i]+newxdatas[i+2])/2, (newydatas[i]+newydatas[i+2])/2

            ax2.plot(newxdatas, newydatas,linewidth = 1)

            
            for i,v in enumerate(newxdatas):
                # print(newxdatas)
                start_point = POINT(newxdatas[i], shape[0]-newydatas[i], 0)
                try:
                    stop_point = POINT(newxdatas[i+1], shape[0]-newydatas[i+1], 0)
                except IndexError :
                    continue
                if AUTOCAD_ENABLE:
                    print('写入到AUTOCAD')
                    a.ActiveDocument.ModelSpace.AddLine(start_point, stop_point)
        #     break
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

if __name__ == '__main__':
    main()