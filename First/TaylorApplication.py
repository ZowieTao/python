# coding: utf-8
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math


def calc_e_small(x):
    n = 10
    f = np.arange(1, n+1).cumprod()
    b = np.array([x]*n).cumprod()
    return np.sum(b/f) + 1


def calc_e(x):
    reverse = False
    if x < 0:
        x = -x
        reverse = True
    ln2 = 0.69314718055994530941723212145818
    c = x/ln2
    a = int(c+0.5)
    b = x - a*ln2
    y = (2 ** a) * calc_e_small(b)
    if reverse:
        return 1/y
    return y


if __name__ == "__main__":
    t1 = np.linspace(-2, 0, 10, endpoint=False)
    t2 = np.linspace(0, 2, 20)
    t = np.concatenate((t1, t2))
    print(t)  # 横轴数据
    y = np.empty_like(t)
    for i, x in enumerate(t):
        y[i] = calc_e(x)
        print('e^', x, '=', y[i], '(近似值)\t', math.exp(x))
        # print '误差：',y[i]-math.exp(x)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y, 'r-', t, y, 'go', linewidth=2)
    plt.title(u'Taylor展开的应用', fontsize=18)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('exp(X)', fontsize=15)
    plt.grid(True)
    plt.show()


# %%
def calc_sin_small(x):
    x2 = -x ** 2
    t = x
    f = 1
    sum = 0
    for i in range(10):
        sum += t/f
        t *= x2
        f *= ((2*i + 2)*(2*i + 3))
        return sum


def calc_sin(x):
    a = x / (2*np.pi)
    k = np.floor(a)
    a = x - k*2*np.pi
    return calc_sin_small(a)


if __name__ == "__main__":
    t = np.linspace(-2*np.pi, 2*np.pi)
