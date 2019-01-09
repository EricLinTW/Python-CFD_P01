#coding=utf-8
###热传与数值分析作业
###2D热扩散方程
###向前差分
from mpl_toolkits.mplot3d import Axes3D ##载入3D绘图
from matplotlib import pyplot, cm
import os
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.inf)
###初始條件
nx = 301 #x网格划分
ny = 301 #y网格划分
nt = 0 #时间步长数
nu = 0.05 #粘度值
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .25
dt = sigma * dx* dy / nu  # 每个时间步长覆盖的时间量

x = np.linspace(0, 5, nx) 
y = np.linspace(0, 5, ny)

u = np.ones((ny, nx)) ##创建初始化的数组 u = 1
un = np.ones((ny, nx)) ##创建迭代数组的预占用保留

###初始状态设定
##设置求解函数的初始状态 I.C. : 设定 u 在 x,y 的 0.5~1 的区间中数值为 8
u[int(0.5 / dy):int(1 / dy + 1),int(0.5 / dx):int(1 / dx + 1)] = 8
 

###初始状态的图形化显示
##3D显示
fig = plt.figure(num='初始状态 3D',figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u, cmap=cm.viridis, rstride=1, cstride=1, linewidth=0, antialiased=False)
##2D显示
plt.figure(num='初始状态 2D')
im=plt.imshow(u,interpolation='bilinear',origin='lower',cmap=cm.rainbow)
plt.contour(X,Y,u)
plt.colorbar(im,orientation='vertical')
##ax.plot_surface(X, Y, v[:], cmap=cm.viridis, rstride=1, cstride=1)
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_zlim(1, 10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$');
##运行步长循环nt个
def diffuse(nt):
    u[int(0.5 / dy):int(1 / dy + 1),int(0.5 / dx):int(1 / dx + 1)] = 8 #函数内定义数值为8的区间
    for n in range (nt+1):
        un = u.copy() # 将u值复制到un
        u[1:-1, 1:-1] = (un[1:-1,1:-1] + 
                        nu * dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        nu * dt / dy**2 * 
                        (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
    ##展示步长结果 3D
    fig = plt.figure(num=' %(nt)d 步长结果 3D' % {'nt':nt},figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=True)
    ax.set_zlim(1, 10)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');
    ##展示步长结果 2D
    plt.figure(num='%(nt)d 步长结果 2D' %  {'nt':nt})
    im=plt.imshow(u,interpolation='bilinear',origin='lower',cmap=cm.rainbow)
    plt.contour(X,Y,u)
    plt.colorbar(im,orientation='vertical')
    ProcessTime = dt*nt
    print("%(nt)d 运行步长(nt) " %  {'nt':nt})
    print("dt= %(dt)f " %  {'dt':dt})
    print("nt= %(nt)f " %  {'nt':nt})
    print("ProcessTime = %(ProcessTime)f " % {'ProcessTime':ProcessTime})


all_ProcessTime = float(input('请输入想要运行的总时间步长(必须是正整数):'))
if all_ProcessTime.is_integer():
    user_Point = float(input('请输入时间步长内想切分的观察点数目(必须是正整数):'))
    if user_Point.is_integer():
        k = int(all_ProcessTime) / int(user_Point)
        if k.is_integer():
            M = int(all_ProcessTime)/int(user_Point)
            for i in range(int(user_Point)):
                i = i+1 
                diffuse(int(M*i))
            plt.show()
        else:
            print('总时长无法按照观察点分割为整数,将观察最接近整数点')
            M = int(all_ProcessTime)/int(user_Point)
            for i in range(int(user_Point)):
                i = i+1 
                diffuse(int(M*i))
            plt.show()
    else:
        print('请输入整数观察点数目')
        os.system("pause")
else:
    print('请输入整数时间步长')
    os.system("pause")




