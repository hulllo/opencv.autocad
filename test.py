import matplotlib.pyplot as plt   
#引入绘图模块  
x=[3,2,5,-8,-3,-4,-5,4,8,3,12,-5,-6,-8]  
y=[-11,-8,6,4,8,6,7,8,21,-3,-5,-32,-15,-40]  
#随机给一些点，这里可以是向量，或者其他值  
z=[]  
for i in range(len(x)):  
    z.append([x[i],y[i]])  
#把x,y坐标点合并，不用zip是因为 zip的结果是tuple,不利于后面的排序，这里需要返回二维的list  
x1=[]  
x2=[]  
x3=[]  
x4=[]  
for i in range(len(z)):  
    if z[i][0]>0 and z[i][1]>0:  
        x1.append(z[i])  
    elif z[i][0]<0 and z[i][1]>0:  
        x2.append(z[i])  
    elif z[i][0]<0 and z[i][1]<0:  
        x3.append(z[i])  
    elif z[i][0]>0 and z[i][1]<0:  
        x4.append(z[i])  
#把坐标点按照四个象限分开存储。如果一开始不是坐标值而是角度或者弧度(如x=[30,60,120..]),这一步  
#就可以省略了  
xx=[]  
x1.sort(key=lambda x:(x[1]/x[0]))  
x2.sort(key=lambda x:(x[1]/x[0]))  
x3.sort(key=lambda x:(x[1]/x[0]))  
x4.sort(key=lambda x:(x[1]/x[0]))  
#依次排序，一开始是想用向量内积的方式得到余弦值在通过反余弦函数求得角度排序，后来发现这样更麻烦  
#实际上 x[1]/x[0] 也可以认为是正切函数值了 2333  
xx=x1+x2+x3+x4  
u=[]  
v=[]  
for i in range(len(xx)):  
    u.append(xx[i][0])  
    v.append(xx[i][1])  
#把 x y再次分开存储。目的是后面作图传递参数方便(也可以不用，只是参数传递不好看)  
u.append(u[0])  
v.append(v[0])  
#在结尾追加第一个坐标值，这样线就闭合了  
#后面的代码就是画图了。。。。。  
fig=plt.figure(figsize=(4,4))  
ax=fig.add_subplot(111)  
ax.spines['top'].set_color('none')  
ax.spines['right'].set_color('none')  
ax.xaxis.set_ticks_position('bottom')  
ax.spines['bottom'].set_position(('data',0))  
#ax.xaxis.set_ticks_position('left')  
#这一步不注视就要报错，哪位大神能不能解释一下啊  
ax.spines['left'].set_position(('data',0))  
plt.plot(u,v)  
plt.scatter(u,v)  
plt.show()  