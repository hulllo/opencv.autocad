import cv2
import numpy as np
def preprocess(filename):

    image = cv2.imread(filename)
    image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
#    直方图均衡增强
    image_equal = cv2.equalizeHist(image_gray)
    r,g,b = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
    
#   拉普拉斯算法增强
    kernel = np.array([ [0, -1, 0],  
                    [-1,  5, -1],  
                    [0, -1, 0] ]) 
    image_lap = cv2.filter2D(image,cv2.CV_8UC3 , kernel)

    cv2.imwrite('image_lap.jpg', image_lap)
#    对象算法增强
    image_log = np.uint8(np.log(np.array(image) +1))    
    cv2.normalize(image_log, image_log,0,255,cv2.NORM_MINMAX)
#    转换成8bit图像显示
    cv2.convertScaleAbs(image_log,image_log)

#    伽马变换
    fgamma = 2
    image_gamma = np.uint8(np.power((np.array(image)/255.0),fgamma)*255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    cv2.imwrite('image_gamma.jpg', image_gamma)


preprocess('2.jpg')    