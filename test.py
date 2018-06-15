# More: http://ex2tron.top

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

# 1.模板匹配
img = cv2.imread('minirealpcb1.jpg')
drawing = np.zeros(img.shape[:], dtype=np.uint8)  # 创建画板
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5) #中值滤波
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh[thresh == 255] = 1
edges = skeletonize(thresh)
# edges = cv2.Canny(thresh, 50, 150)
# 2.统计概率霍夫线变换
drawing = np.zeros(img.shape[:], dtype=np.uint8)
edges = edges.astype(np.uint8)*255
print(edges.shape, gray.shape)
lines = cv2.HoughLinesP(edges, 5, np.pi / 4, 10,
                        minLineLength=3, maxLineGap=15)
# lines = cv2.HoughLines(edges, 0.8, np.pi / 4, 6)
                        
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,
                         sharex=True, sharey=True)
ax2.axis('image')
# 将检测的线画出来
newxs = []
newys = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    newxs.append([x1,x2])
    newys.append([y1,y2])
    # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
ax2.imshow(edges,cmap = 'gray')    
for i,v in enumerate(newxs):    
    ax2.plot(newxs[i], newys[i])

# cv2.imshow('probabilistic hough lines', np.hstack((img, img)))
# cv2.waitKey(0)

ax1.imshow(edges)
plt.show()
# # 3.霍夫圆变换
# drawing = np.zeros(img.shape[:], dtype=np.uint8)

# circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param2=30)
# circles = np.int0(np.around(circles))

# # 将检测的圆画出来
# for i in circles[0, :]:
#     cv2.circle(drawing, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画出外圆
#     cv2.circle(drawing, (i[0], i[1]), 2, (0, 0, 255), 3)  # 画出圆心

# cv2.imshow('circles', np.hstack((img, drawing)))
# cv2.waitKey(0)
