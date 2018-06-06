import numpy as np
data = np.sin(np.arange(20)).reshape(5, 4)
print (data)
ind = data.argmax(axis=0) # 按列得到每一列中最大元素的索引，axis=1为按行
print (ind)
data_max = data[ind, range(data.shape[1])] # 将最大值取出来
# print (data_max)