from numpy import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.animation as animation

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

# print max and min
loss_data_fin_3d.flatten()
print(loss_data_fin_3d.shape)
print(loss_data_fin_3d[0][0][0])

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

# plot loss values of 3d loss landscape
num_list = []
for i in range(STEPS):
    for j in range(STEPS):
        for k in range(STEPS):
            num_list.append(loss_data_fin_3d[i][j][k])
plt.bar(range(len(num_list)), num_list)
plt.yticks(np.arange(2.0, 2.5, step=0.01))
plt.ylim(ymin=2.0, ymax=2.5)
plt.savefig('loss_mnist_3d_values.png')
plt.show()

plt.plot(range(len(num_list)), num_list,'o-',color = 'r',label="loss")
plt.savefig('loss_mnist_3d_values_line.png')
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

loss_data_fin.flatten()
print('max loss fin', np.max(loss_data_fin))
print('min loss fin', np.min(loss_data_fin))

# plot loss values of 2d loss landscape
num_list_2d = []
for i in range(STEPS):
    for j in range(STEPS):
            num_list_2d.append(loss_data_fin[i][j])
plt.bar(range(len(num_list_2d)), num_list_2d)
plt.yticks(np.arange(2.0, 2.5, step=0.01))
plt.ylim(ymin=2.0, ymax=2.5)
plt.savefig('loss_mnist_2d_values.png')
plt.show()