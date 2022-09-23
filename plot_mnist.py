from numpy import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm

STEPS = 40

# load array
X = load('X.npy')
Y = load('Y.npy')
Z = load('Z.npy')
loss_data_fin = load('loss_data_fin.npy')
loss_data_fin_3d = load('loss_data_fin_3d.npy')

# print the array
print(X.shape)
print(Y.shape)
print(Z.shape)
print(loss_data_fin.shape)
print(loss_data_fin_3d.shape)
print(loss_data_fin_3d)

# print max and min
loss_data_fin_3d.flatten()
print('max loss fin', np.max(loss_data_fin_3d))
print('min loss fin', np.min(loss_data_fin_3d))

# plot loss landscape 3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, Z, c=loss_data_fin_3d, cmap='rainbow')

# add plot labels
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'black'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'black'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'black'})

# save plot to file and show
plt.savefig('loss_mnist_3d_plot.png')
plt.show()

# plot loss contour in 2D
fig = plt.figure()
ax = plt.axes(projection='3d')
X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

cset = ax.contourf(X, Y, loss_data_fin, zdir='z', offset=3, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, loss_data_fin, zdir='x', offset=40, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, loss_data_fin, zdir='y', offset=0, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(0, 40)
ax.set_ylabel('Y')
ax.set_ylim(0, 40)
ax.set_zlabel('loss_data_fin')
ax.set_zlim(2, 3)

ax.set_title('Surface and Contour Plot of Loss Landscape')

# save plot to file and show
plt.savefig('loss_mnist_2d_plot.png')
plt.show()