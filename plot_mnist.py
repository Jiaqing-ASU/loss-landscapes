from numpy import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.animation as animation

STEPS = 40

# load array
X = load('mnist_results/X.npy')
Y = load('mnist_results/Y.npy')
Z = load('mnist_results/Z.npy')
loss_data_fin = load('mnist_results/loss_data_fin_mnist.npy')
loss_data_fin_3d = load('mnist_results/loss_data_fin_3d_mnist.npy')
loss_data_fin_c = load('mnistc_results/loss_data_fin_mnistc.npy')
loss_data_fin_3d_c = load('mnistc_results/loss_data_fin_3d_mnistc.npy')

# print the array
print(X.shape)
print(Y.shape)
print(Z.shape)
print(loss_data_fin.shape)
print(loss_data_fin_3d.shape)
print(loss_data_fin_c.shape)
print(loss_data_fin_3d_c.shape)

# print max and min
loss_data_fin_3d.flatten()
print('max loss fin', np.max(loss_data_fin_3d))
print('min loss fin', np.min(loss_data_fin_3d))
print(loss_data_fin_3d.shape)

loss_data_fin.flatten()
print('max loss fin', np.max(loss_data_fin))
print('min loss fin', np.min(loss_data_fin))
print(loss_data_fin.shape)

loss_data_fin_3d_c.flatten()
print('max loss fin', np.max(loss_data_fin_3d_c))
print('min loss fin', np.min(loss_data_fin_3d_c))
print(loss_data_fin_3d_c.shape)

loss_data_fin_c.flatten()
print('max loss fin', np.max(loss_data_fin_c))
print('min loss fin', np.min(loss_data_fin_c))
print(loss_data_fin_c.shape)

# plot loss landscape 3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, Z, c=loss_data_fin_3d, cmap='rainbow')

# add plot labels
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'black'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'black'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'black'})

# save plot to file and show
plt.savefig('mnist_results/loss_mnist_3d_plot.png')
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
plt.savefig('mnist_results/loss_mnist_3d_values.png')
plt.show()

plt.plot(range(len(num_list)), num_list,'o-',color = 'r',label="loss")
plt.savefig('mnist_results/loss_mnist_3d_values_line.png')
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
plt.savefig('mnist_results/loss_mnist_2d_plot.png')
plt.show()

# plot loss values of 2d loss landscape
num_list_2d = []
for i in range(STEPS):
    for j in range(STEPS):
            num_list_2d.append(loss_data_fin[i][j])
plt.bar(range(len(num_list_2d)), num_list_2d)
plt.yticks(np.arange(2.0, 2.5, step=0.01))
plt.ylim(ymin=2.0, ymax=2.5)
plt.savefig('mnist_results/loss_mnist_2d_values.png')
plt.show()

# plot the mnist-c data

# prepare data for plotting
X_list_C = []
Y_list_C = []
Z_list_C = []

for i in range(0, STEPS):
    for j in range(0, STEPS):
        for k in range(0, STEPS):
            X_list_C.append(i)
            Y_list_C.append(j)
            Z_list_C.append(k)

X_C = np.array(X_list_C)
Y_C = np.array(Y_list_C)
Z_C = np.array(Z_list_C)

# plot loss landscape 3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_C, Y_C, Z_C, c=loss_data_fin_3d_c, cmap='rainbow')

# add plot labels
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'black'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'black'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'black'})

# save plot to file and show
plt.savefig('mnistc_results/loss_mnistc_3d_plot.png')
plt.show()

# plot loss values of 3d loss landscape
num_list_c = []
for i in range(STEPS):
    for j in range(STEPS):
        for k in range(STEPS):
            num_list_c.append(loss_data_fin_3d_c[i][j][k])
plt.bar(range(len(num_list_c)), num_list_c)
plt.yticks(np.arange(2.0, 2.5, step=0.01))
plt.ylim(ymin=2.0, ymax=2.5)
plt.savefig('mnistc_results/loss_mnistc_3d_values.png')
plt.show()

plt.plot(range(len(num_list_c)), num_list_c,'o-',color = 'r',label="loss")
plt.savefig('mnistc_results/loss_mnistc_3d_values_line.png')
plt.show()

# plot loss contour in 2D
fig = plt.figure()
ax = plt.axes(projection='3d')
X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
ax.plot_surface(X, Y, loss_data_fin_c, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

cset = ax.contourf(X, Y, loss_data_fin_c, zdir='z', offset=3, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, loss_data_fin_c, zdir='x', offset=40, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, loss_data_fin_c, zdir='y', offset=0, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(0, 40)
ax.set_ylabel('Y')
ax.set_ylim(0, 40)
ax.set_zlabel('loss_data_fin')
ax.set_zlim(2, 3)

ax.set_title('Surface and Contour Plot of Loss Landscape')

# save plot to file and show
plt.savefig('mnistc_results/loss_mnistc_2d_plot.png')
plt.show()

# plot loss values of 2d loss landscape
num_list_2d_c = []
for i in range(STEPS):
    for j in range(STEPS):
            num_list_2d_c.append(loss_data_fin_c[i][j])
plt.bar(range(len(num_list_2d_c)), num_list_2d_c)
plt.yticks(np.arange(2.0, 2.5, step=0.01))
plt.ylim(ymin=2.0, ymax=2.5)
plt.savefig('mnistc_results/loss_mnistc_2d_values.png')
plt.show()