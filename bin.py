import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

def track_back(x):
    pass

path=os.path.dirname(__file__)
img = cv2.imread(path+'/2.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('thresh',0)
cv2.createTrackbar('param1', 'thresh', 11, 200, track_back)
cv2.createTrackbar('param2', 'thresh', 4, 255, track_back)
while(True):
    param1 = cv2.getTrackbarPos('param1', 'thresh')
    param2 = cv2.getTrackbarPos('param2', 'thresh')        # 固定阈值
    # ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # 自适应阈值
    if param1%2 != 1:
        param1 = param1 + 1
    th2 = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, param1, param2)
    # th3 = cv2.adaptiveThreshold(
        # img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
    cv2.imshow('thresh', th2)
    if cv2.waitKey(30) == 27:
        break
