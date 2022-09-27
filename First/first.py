# coding: utf-8
# %%
from cProfile import label
from cgi import print_arguments
from re import T
from traceback import print_tb
from matplotlib import rcParams
from symbol import parameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import gauss
from scipy import stats
from sklearn.cluster import KMeans
import math
from PIL import Image
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture
# import cv2
import matplotlib


# Code exp 1
# if __name__ == "__main__":
#     x = [float(i)/100.0 for i in range(1, 300)]
#     y = [math.log(i) for i in x]
#     plt.plot(x, y, 'r-', linewidth=3, label='log Curve')
#     a = [x[20], x[175]]
#     b = [y[20], y[175]]
#     plt.plot(a, b, 'g-', linewidth=2, label="line")
#     plt.legend(loc='upper left')
#     plt.grid(True)
#     plt.xlabel('X')
#     plt.ylabel('log(X)')
#     plt.show()

# Code exp 2
# if __name__ == "__main__":
#     u = np.random.uniform(0.0, 1.0, 10000)
#     plt.hist(u, 80, facecolor='g', alpha=0.75)
#     plt.grid(True)
#     plt.show()

#     times = 10000
#     for time in range(times):
#         u += np.random.uniform(0.0, 1.0, 10000)
#     print(len(u))
#     plt.hist(u, 80, facecolor='g', alpha=0.75)
#     plt.grid(True)
#     plt.show()

# # EM Code
# def calcM(height):
#     N = len(height)
#     gp = 0.5  # girl probability
#     bp = 0.5  # boy probability
#     gmu, gsigma = min(height), 1
#     bmu, bsigma = max(height), 1
#     ggamma = range(N)
#     bgamma = range(N)
#     cur = [gp, bp, gmu, gsigma, bmu, bsigma]
#     now = []

#     times = 0
#     while times < 100:
#         i = 0
#         for x in height:
#             ggamma[i] = gp * gauss(x, gmu, gsigma)
#             bgamma[i] = bp * gauss(x, bmu, bsigma)
#             s = ggamma[i] + bgamma[i]
#             ggamma[i] /= s
#             bgamma[i] /= s
#             i += 1

#         gn = sum(ggamma)
#         gp = float(gn) / float(N)
#         bn = sum(bgamma)
#         bp = float(bn) / float(N)
#         gmu = averageWeight(height, ggamma, gn)
#         gsigma = varianceWeight(height, ggamma, gn)
#         bmu = averageWeight(height, bgamma, bn)
#         gsigma = varianceWeight(height, bgamma, bmu, bn)

#         now = [gp, bp, gmu, gsigma, bmu, bsigma]
#         if isSame(cur, now):
#             break
#         cur = now
#         print('Tiems: \t', times)
#         print('Girl mean/gsigma \t ', gmu, gsigma)
#         print('Boy mean/bsigma \t ', bmu, bsigma)
#         print('Boy/Girl \t ', bn, gn, bn+gn)
#         print('\n\n')
#         times += 1
#     return now

# # GMM and image
# def composite(band, parameter):
#     c1 = parameter[0]
#     mu1 = parameter[2]
#     sigma1 = parameter[3]
#     c2 = parameter[1]
#     mu2 = parameter[4]
#     sigma2 = parameter[5]

#     p1 = []
#     p2 = []
#     for pixel in band:
#         p1.append(c1 * gauss(pixel, mu1, sigma1))
#         p2.append(c2 * gauss(pixel, mu2, sigma2))

#     scale(p1)  # 灰度均衡
#     scale(p2)
#     return [p1, p2]


# if __name__ == "__main__":
#     im = Image.open('son.png')
#     print(im.format, im.size, im.mode)

#     im = im.split()[0]  # 只处理第一个通道
#     nb = []  # 处理后的新通道
#     data = list(im.getdata())
#     # print(data)
#     parameter = GMM(data)
#     t = composite(data, parameter)

#     im1 = Image.new('L', im.size)
#     im1.putdata(t[0])


# # Taylor application 1 ： calc e^x value
# def calc_e_small(x):
#     n = 10
#     f = np.arange(1, n+1).cumprod()
#     b = np.array([x]*n).cumprod()
#     return np.sum(b/f) + 1


# def calc_e(x):
#     reverse = False
#     if x < 0:
#         x = -x
#         reverse = True
#     ln2 = 0.69314718055994530941723212145818
#     c = x/ln2
#     a = int(c+0.5)
#     b = x - a*ln2
#     y = (2 ** a) * calc_e_small(b)
#     if reverse:
#         return 1/y
#     return y


# if __name__ == "__main__":
#     t1 = np.linspace(-2, 0, 10, endpoint=False)
#     t2 = np.linspace(0, 2, 20)
#     t = np.concatenate((t1, t2))
#     print(t)  # 横轴数据
#     y = np.empty_like(t)
#     for i, x in enumerate(t):
#         y[i] = calc_e(x)
#         print('e^', x, '=', y[i], '(近似值)\t', math.exp(x))
#         # print '误差：',y[i]-math.exp(x)
#     plt.plot(t, y, 'r-', t, y, 'go', linewidth=2)
#     plt.title(u'Taylor展开的应用', fontsize=18)
#     plt.xlabel('x', fontsize=15)
#     plt.ylabel('exp(X)', fontsize=15)
#     plt.grid(True)
#     plt.show()

# # Taylor application 2 ： calc_sin
# def calc_sin_small(x):
#     x2 = -x ** 2
#     t = x
#     f = 1
#     sum = 0
#     for i in range(10):
#         sum += t / f
#         t *= x2
#         f *= ((2*i+2)*(2*i+3))
#     return sum


# def calc_sin(x):
#     a = x / (2*np.pi)
#     k = np.floor(a)
#     a = x - k*2*np.pi
#     return calc_sin_small(a)


# if __name__ == "__main__":
#     t = np.linspace(-2*np.pi, 2*np.pi, 100, endpoint=False)
#     print(t)     # 横轴数据
#     y = np.empty_like(t)
#     for i, x in enumerate(t):
#         y[i] = calc_sin(x)
#         print('sin(', x, ') = ', y[i], '(近似值)\t', math.sin(x), '(真实值)')
#         print('误差：', y[i] - math.sin(x))
#     plt.figure(facecolor='w')
#     plt.plot(t, y, 'r-', t, y, 'go', linewidth=2)
#     plt.title('Taylor展式的应用 - 正弦函数', fontsize=18)
#     plt.xlabel('X', fontsize=15)
#     plt.ylabel('sin(X)', fontsize=15)
#     plt.xlim((-7, 7))
#     plt.ylim((-1.1, 1.1))
#     plt.grid(True)
#     plt.show()


# Gini 系数的生成
if __name__ == "__main__":
    p = np.arange(0.001, 1, 0.001, dtype=np.float)
    gini = 2 * p * (1-p)
    h = -(p * np.log2(p) + (1-p) * np.log2(1-p))/2
    err = 1 - np.max(np.vstack((p, 1-p)), 0)
    plt.plot(p, h, 'b-', linewidth=2, label="Entropy")
    plt.plot(p, gini, 'r-', linewidth=2, label="Gini")
    plt.plot(p, err, 'g-.', linewidth=1, label="Error")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()


# data= 2*np.random.rand(100000,2)-1
# print(data)
# x=(data[:,0])
# y=(data[:,1])
# idx= x**2+ y**2 < 1
# hole= x**2+ y**2 < 0.25
# idx = np.logical_and(idx,~hole)
# plt.plot(x[idx],y[idx],'go',markersize=1)


# p = np.random.rand(10000)
# np.set_printoptions(edgeitems=5000,suppress=True)
# plt.hist(p,bins=20,color="g",edgecolor="k")
# plt.show()

# N = 10000
# times = 100
# z = np.zeros(N)
# for i in range(times):
#   z += np.random.rand(N)
# z /= times
# plt.hist(z,bins=20,color='m',edgecolor='k')
# plt.show()

# d = np.random.rand(3,4)
# print(d)
# print(type(d))
# data = pd.DataFrame(data = d,columns=list('梅兰竹菊'))
# print('='*50)
# print(data)
# print(type(data))
# print(data[list('兰菊')])
# data.to_csv('data.csv',index=False,header=True)
# print('file save success')

# # [-4,2]
# d= np.random.rand(100)*6 - 4
# print(d)
# plt.plot(d,'r.')
# plt.show()

# # x = np.arange(0,1,0.000001)
# x = np.linspace(0,1,10)
# print(x)
# y = x ** x
# plt.plot(x,y,'r-',linewidth=3)
# plt.show()


# x = np.arange(0.05,3,0.05)
# y1 = [math.log(a,1.5) for a in x]
# plt.plot(x,y1,linewidth=2,color="#007500", label="log1.5(x)")
# y2 = [math.log(a,2) for a in x]
# plt.plot(x,y2,linewidth=2,color="#9f35ff",label="log2(x)")
# y3 = [math.log(a,3)for a in x]
# plt.plot(x,y3,linewidth=2,color="#f75000",label="log3(x)")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()

# image = cv2.imread('lena.png')
# print(image)
# print(type(image))
# print(image.shape)


# a = Image.open('lena.png')
# print(a)
# b = np.array(a)
# print(b)
# print(type(b))

# %%
