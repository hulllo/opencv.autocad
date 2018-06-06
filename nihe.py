import numpy as np  
import matplotlib.pyplot as plt  
from scipy import optimize  
  
#直线方程函数  
def f_1(x, A, B):  
    return A*x + B  
  
#二次曲线方程  
def f_2(x, A, B, C):  
    return A*x*x + B*x + C  
  
#三次曲线方程  
def f_3(x, A, B, C, D):  
    return A*x*x*x + B*x*x + C*x + D  
  
def plot_test(x0,y0):  
  
    plt.figure()  
  
    #拟合点  
    # x0 = [1, 2, 3, 4, 5]  
    # y0 = [1, 3, 8, 18, 36]  
  
    #绘制散点  
    # plt.scatter(x0[:], y0[:], 25, "red")  
  
    #直线拟合与绘制  
    A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]  
    x1 = np.arange(0, 6, 0.01)  
    y1 = A1*x1 + B1  
    plt.plot(x1, y1, "blue")  
  
    #二次曲线拟合与绘制  
    A2, B2, C2 = optimize.curve_fit(f_2, x0, y0)[0]  
    x2 = np.arange(0, 6, 0.01)  
    y2 = A2*x2*x2 + B2*x2 + C2   
    plt.plot(x2, y2, "green")  
  
    #三次曲线拟合与绘制  
    A3, B3, C3, D3= optimize.curve_fit(f_3, x0, y0)[0]  
    x3 = np.arange(0, 6, 0.01)  
    y3 = A3*x3*x3*x3 + B3*x3*x3 + C3*x3 + D3   
    plt.plot(x3, y3, "purple")  
  
    plt.title("test")  
    plt.xlabel('x')  
    plt.ylabel('y')  
  
    plt.show()  
  
    return  

if __name__ == '__main__':
    x0 = [1, 2, 3, 4, 5]  
    y0 = [1, 3, 8, 18, 36]  
    plot_test(x0,y0)