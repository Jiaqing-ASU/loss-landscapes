from numpy import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.animation as animation

STEPS = 40

# load array
loss_data_fin = load('cifar10_results/loss_data_fin_cifar10.npy')
loss_data_fin_3d = load('cifar10_results/loss_data_fin_3d_cifar10.npy')
loss_data_fin_c = load('cifar10c_results/loss_data_fin_cifar10c.npy')
loss_data_fin_3d_c = load('cifar10c_results/loss_data_fin_3d_cifar10c.npy')

# print the array
print(loss_data_fin.shape)
print(loss_data_fin_3d.shape)
print(loss_data_fin_c.shape)
print(loss_data_fin_3d_c.shape)

# print max and min
loss_data_fin_3d.flatten()
print('max loss fin', np.max(loss_data_fin_3d))
print('min loss fin', np.min(loss_data_fin_3d))

loss_data_fin.flatten()
print('max loss fin', np.max(loss_data_fin))
print('min loss fin', np.min(loss_data_fin))

loss_data_fin_3d_c.flatten()
print('max loss fin', np.max(loss_data_fin_3d_c))
print('min loss fin', np.min(loss_data_fin_3d_c))

loss_data_fin_c.flatten()
print('max loss fin', np.max(loss_data_fin_c))
print('min loss fin', np.min(loss_data_fin_c))

# plot loss values of 3d loss landscape of cifar10
num_list = []
for i in range(STEPS):
    for j in range(STEPS):
        for k in range(STEPS):
            num_list.append(loss_data_fin_3d[i][j][k])
plt.bar(range(len(num_list)), num_list)
#plt.yticks(np.arange(2.0, 2.5, step=0.01))
#plt.ylim(ymin=2.0, ymax=2.5)
plt.savefig('cifar10_results/loss_cifar10_3d_values.png')
plt.show()

# plot loss values of 3d loss landscape of cifar10c
num_list_c = []
for i in range(STEPS):
    for j in range(STEPS):
        for k in range(STEPS):
            num_list_c.append(loss_data_fin_3d_c[i][j][k])
plt.bar(range(len(num_list_c)), num_list_c)
#plt.yticks(np.arange(2.0, 2.5, step=0.01))
#plt.ylim(ymin=2.0, ymax=2.5)
plt.savefig('cifar10c_results/loss_cifar10c_3d_values.png')
plt.show()

# plot loss values of 2d loss landscape of cifar10
num_list_2d = []
for i in range(STEPS):
    for j in range(STEPS):
            num_list_2d.append(loss_data_fin[i][j])
plt.bar(range(len(num_list_2d)), num_list_2d)
#plt.yticks(np.arange(2.0, 2.5, step=0.01))
#plt.ylim(ymin=2.0, ymax=2.5)
plt.savefig('cifar10_results/loss_cifar10_2d_values.png')
plt.show()

# plot loss values of 2d loss landscape of cifar10c
num_list_2d_c = []
for i in range(STEPS):
    for j in range(STEPS):
            num_list_2d_c.append(loss_data_fin_c[i][j])
plt.bar(range(len(num_list_2d_c)), num_list_2d_c)
#plt.yticks(np.arange(2.0, 2.5, step=0.01))
#plt.ylim(ymin=2.0, ymax=2.5)
plt.savefig('cifar10c_results/loss_cifar10c_2d_values.png')
plt.show()
