import cv2
import matplotlib.pyplot as plt
import os
path=os.path.dirname(__file__)

img = cv2.imread(path+'/realpcb1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 闭运算
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
thresh = cv2.medianBlur(thresh, 5)  # 中值滤波
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)
fig, (ax1, ax2) = plt.subplots(ncols = 2, sharex = True, sharey = True)

ax1.imshow(thresh,cmap = 'gray')
ax1.axis('off')

ax2.axis('image')
ax2.set_xticks([])
ax2.set_yticks([])
for n, contour in enumerate(contours):
    xdata = contour[:,:,0]
    ydata = contour[:,:,1]
    ax2.plot(xdata,ydata)
plt.show()