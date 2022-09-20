# coding:utf-8
#%%
import numpy as np
import matplotlib.pyplot as plt

x= np.linspace(0.0001,2,200)
y= x**x
plt.plot(x,y,'r-',mec='k',linewidth=3)
plt.show()


# %%
