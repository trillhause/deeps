import numpy as np
import matplotlib.pyplot as plt

speed = np.loadtxt('data/train.txt')
speed = speed.reshape(1,speed.shape[0])

time = np.arange(0,1020,0.05)
time = time.reshape(1,20400)

print(time.shape)
# plt.plot([1,2,3],[4,5,6])
# plt.show()