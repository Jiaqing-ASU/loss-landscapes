from numpy import load
import numpy as np

STEPS = 40

# load array
X = load('mnist_results/X.npy')
Y = load('mnist_results/Y.npy')
Z = load('mnist_results/Z.npy')
loss_data_fin_3d = load('mnist_results/loss_data_fin_3d_mnist.npy')
loss_data_fin = load('mnist_results/loss_data_fin_mnist.npy')
loss_list = []

for i in range(0, STEPS):
    for j in range(0, STEPS):
        for k in range(0, STEPS):
            loss_list.append(loss_data_fin_3d[i][j][k])

loss = np.array(loss_list)

# print the array
print(X.shape)
print(Y.shape)
print(Z.shape)
print(loss.shape)

point_list = []

for i in range(0, STEPS*STEPS*STEPS):
    point = []
    point.append(X[i])
    point.append(Y[i])
    point.append(Z[i])
    point.append(loss[i])
    point_list.append(point)

np.savetxt("points/points.csv", point_list, delimiter=",")

point_raw = []

for i in range(0, STEPS*STEPS*STEPS):
    point_raw.append(X[i])
    point_raw.append(Y[i])
    point_raw.append(Z[i])
    point_raw.append(loss[i])

np.array(point_raw,np.float32).tofile("points/points.raw")

test_array = np.arange(100).reshape(5,20).astype(np.float)
print(test_array)

np.savetxt("points/points_test.csv", test_array, delimiter=",")
np.array(test_array,np.float32).tofile("points/points_test.raw")

# export the loss values for 3d
np.array(loss_data_fin_3d,np.float32).tofile("points/points_loss_3d.raw")

# export the loss values for 2d
np.array(loss_data_fin,np.float32).tofile("points/points_loss_2d.raw")