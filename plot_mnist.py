from numpy import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.animation as animation
import argparse
import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
from numpy import save

matplotlib.rcParams['figure.figsize'] = [18, 12]

# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics

# training hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2
BATCH_SIZE = 512
EPOCHS = 25
# contour plot resolution
STEPS = 40

class MLPSmall(torch.nn.Module):
    """ Fully connected feed-forward neural network with one hidden layer. """
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 32)
        self.linear_2 = torch.nn.Linear(32, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        return F.softmax(self.linear_2(h), dim=1)

class Flatten(object):
    """ Transforms a PIL image to a flat numpy array. """
    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()

parser = argparse.ArgumentParser()
# input corruption dataset:
# brightness, canny_edges, dotted_line, fog, glass_blur, identity
# impulse_noise, motion_blur, rotate, scale, shear, shot_noise
# spatter, stripe, translate, zigzag
parser.add_argument('--dataset', default='brightness', help='input corruption dataset')
# input trained model:
# original, brightness, canny_edges, dotted_line, fog, glass_blur, identity
# impulse_noise, motion_blur, rotate, scale, shear, shot_noise
# spatter, stripe, translate, zigzag
parser.add_argument('--model', default='original', help='model dataset')
args = parser.parse_args()
model_path = 'mnist_model' + '/' + args.model + '/'
path = 'mnist_results' + '/' + args.dataset + '/'

# download MNIST and setup data loaders
mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True, transform=Flatten())
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()

# load model
model_initial = MLPSmall(IN_DIM, OUT_DIM)
model_initial.load_state_dict(torch.load(model_path+'model_initial.pt'))
model_initial.eval()
model_final = MLPSmall(IN_DIM, OUT_DIM)
model_final.load_state_dict(torch.load(model_path+'model_final.pt'))
model_final.eval()

# data that the evaluator will use when evaluating loss
x, y = iter(train_loader).__next__()
metric = loss_landscapes.metrics.Loss(criterion, x, y)

# compute loss data
loss_data_original = loss_landscapes.linear_interpolation(model_initial, model_final, metric, STEPS, deepcopy_model=True)

# plot loss data in 1D
plt.plot([1/STEPS * i for i in range(STEPS)], loss_data_original)
plt.title('Linear Interpolation of Loss for ' + args.model + ' model with original dataset')
plt.xlabel('Interpolation Coefficient')
plt.ylabel('Loss')
axes = plt.gca()

# save plot to file and show
plt.savefig('mnist_results/original/loss_mnist_1d_'+args.model+'.png')
plt.show()

loss_data_fin_original, dir_one, dir_two = loss_landscapes.random_plane(model_final, metric, 10, STEPS, normalization='filter', deepcopy_model=True)

# plot loss contour in 2D
plt.contour(loss_data_fin_original, levels=50)
plt.title('Loss Contours around Trained Model for ' + args.model + ' model with original dataset')

# save plot to file and show
plt.savefig('mnist_results/original/loss_mnist_2d_'+args.model+'.png')
plt.show()

# compute loss landscape 3D data
loss_data_fin_3d_original, dir_one_space, dir_two_space, dir_three_space = loss_landscapes.random_space(model_final, metric, 10, STEPS, normalization='filter', deepcopy_model=True)

# reshape loss data for 3D plot
loss_data_fin_3d_original.reshape(-1)

# prepare data for plotting
X_list = []
Y_list = []
Z_list = []

for i in range(0, STEPS):
    for j in range(0, STEPS):
        for k in range(0, STEPS):
            X_list.append(i)
            Y_list.append(j)
            Z_list.append(k)

X = np.array(X_list)
Y = np.array(Y_list)
Z = np.array(Z_list)
 
# plot loss landscape 3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, Z, c=loss_data_fin_3d_original, cmap='rainbow')

# add plot labels
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'black'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'black'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'black'})
plt.title('Loss Landscape in 3D for ' + args.model + ' model with original dataset')

# save plot to file and show
plt.savefig('mnist_results/original/loss_mnist_3d_'+args.model+'.png')
plt.show()

save('mnist_results/original/loss_data_fin_mnist_'+args.model+'.npy', loss_data_fin_original)
save('mnist_results/original/loss_data_fin_3d_mnist_'+args.model+'.npy', loss_data_fin_3d_original)

# data that the evaluator will use when evaluating loss
if args.dataset == 'brightness':
    x_c = load('mnist_c/brightness/train_images.npy')
    y_c = load('mnist_c/brightness/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/brightness/'
elif args.dataset == 'canny_edges':
    x_c = load('mnist_c/canny_edges/train_images.npy')
    y_c = load('mnist_c/canny_edges/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/canny_edges/'
elif args.dataset == 'dotted_line':
    x_c = load('mnist_c/dotted_line/train_images.npy')
    y_c = load('mnist_c/dotted_line/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/dotted_line/'
elif args.dataset == 'fog':
    x_c = load('mnist_c/fog/train_images.npy')
    y_c = load('mnist_c/fog/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/fog/'
elif args.dataset == 'glass_blur':
    x_c = load('mnist_c/glass_blur/train_images.npy')
    y_c = load('mnist_c/glass_blur/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/glass_blur/'
elif args.dataset == 'identity':
    x_c = load('mnist_c/identity/train_images.npy')
    y_c = load('mnist_c/identity/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/identity/'
elif args.dataset == 'impulse_noise':
    x_c = load('mnist_c/impulse_noise/train_images.npy')
    y_c = load('mnist_c/impulse_noise/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/impulse_noise/'
elif args.dataset == 'motion_blur':
    x_c = load('mnist_c/motion_blur/train_images.npy')
    y_c = load('mnist_c/motion_blur/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/motion_blur/'
elif args.dataset == 'rotate':
    x_c = load('mnist_c/rotate/train_images.npy')
    y_c = load('mnist_c/rotate/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/rotate/'
elif args.dataset == 'scale':
    x_c = load('mnist_c/scale/train_images.npy')
    y_c = load('mnist_c/scale/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/scale/'
elif args.dataset == 'shear':
    x_c = load('mnist_c/shear/train_images.npy')
    y_c = load('mnist_c/shear/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/shear/'
elif args.dataset == 'shot_noise':
    x_c = load('mnist_c/shot_noise/train_images.npy')
    y_c = load('mnist_c/shot_noise/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/shot_noise/'
elif args.dataset == 'spatter':
    x_c = load('mnist_c/spatter/train_images.npy')
    y_c = load('mnist_c/spatter/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/spatter/'
elif args.dataset == 'stripe':
    x_c = load('mnist_c/stripe/train_images.npy')
    y_c = load('mnist_c/stripe/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/stripe/'
elif args.dataset == 'translate':
    x_c = load('mnist_c/translate/train_images.npy')
    y_c = load('mnist_c/translate/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/translate/'
elif args.dataset == 'zigzag':
    x_c = load('mnist_c/zigzag/train_images.npy')
    y_c = load('mnist_c/zigzag/train_labels.npy')
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    path = 'mnist_results/zigzag/'
metric_c = loss_landscapes.metrics.Loss(criterion, torch.from_numpy(x_c.reshape(60000, 784)).float(), y_c)

# compute loss data
loss_data_c = loss_landscapes.linear_interpolation(model_initial, model_final, metric, STEPS, deepcopy_model=True)

# plot loss data in 1D
plt.plot([1/STEPS * i for i in range(STEPS)], loss_data_c)
plt.title('Linear Interpolation of Loss for ' + args.model + ' model with ' + args.dataset + ' dataset')
plt.xlabel('Interpolation Coefficient')
plt.ylabel('Loss')
axes = plt.gca()

# save plot to file and show
plt.savefig(path+'loss_mnistc_1d_'+args.model+'.png')
plt.show()

loss_data_fin_c = loss_landscapes.random_plane_given_plane(model_final, metric_c, dir_one, dir_two, 10, STEPS, normalization='filter', deepcopy_model=True)

# plot loss contour in 2D
plt.contour(loss_data_fin_c, levels=50)
plt.title('Loss Contours around Trained Model for ' + args.model + ' model with ' + args.dataset + ' dataset')

# save plot to file and show
plt.savefig(path+'loss_mnistc_2d_'+args.model+'.png')
plt.show()

# compute loss landscape 3D data
loss_data_fin_3d_c = loss_landscapes.random_space_given_space(model_final, metric_c, dir_one_space, dir_two_space, dir_three_space, 10, STEPS, normalization='filter', deepcopy_model=True)

# reshape loss data for 3D plot
loss_data_fin_3d_c.reshape(-1)

# prepare data for plotting
X_list = []
Y_list = []
Z_list = []

for i in range(0, STEPS):
    for j in range(0, STEPS):
        for k in range(0, STEPS):
            X_list.append(i)
            Y_list.append(j)
            Z_list.append(k)

X = np.array(X_list)
Y = np.array(Y_list)
Z = np.array(Z_list)
 
# plot loss landscape 3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, Z, c=loss_data_fin_3d_c, cmap='rainbow')

# add plot labels
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'black'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'black'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'black'})
plt.title('Loss Landscape in 3D for ' + args.model + ' model with ' + args.dataset + ' dataset')

# save plot to file and show
plt.savefig(path+'loss_mnistc_3d_'+args.model+'.png')
plt.show()

save(path+'loss_data_fin_mnistc_'+args.model+'.npy', loss_data_fin_c)
save(path+'loss_data_fin_3d_mnistc_'+args.model+'.npy', loss_data_fin_3d_c)

# print the array
print(X.shape)
print(Y.shape)
print(Z.shape)
print(loss_data_fin_original.shape)
print(loss_data_fin_3d_original.shape)
print(loss_data_fin_c.shape)
print(loss_data_fin_3d_c.shape)

# print max and min
loss_data_fin_3d_original.flatten()
print('max loss fin', np.max(loss_data_fin_3d_original))
print('min loss fin', np.min(loss_data_fin_3d_original))
print(loss_data_fin_3d_original.shape)

loss_data_fin_original.flatten()
print('max loss fin', np.max(loss_data_fin_original))
print('min loss fin', np.min(loss_data_fin_original))
print(loss_data_fin_original.shape)

loss_data_fin_3d_c.flatten()
print('max loss fin', np.max(loss_data_fin_3d_c))
print('min loss fin', np.min(loss_data_fin_3d_c))
print(loss_data_fin_3d_c.shape)

loss_data_fin_c.flatten()
print('max loss fin', np.max(loss_data_fin_c))
print('min loss fin', np.min(loss_data_fin_c))
print(loss_data_fin_c.shape)

# plot loss values of 3d loss landscape
num_list_3d = []
for i in range(STEPS):
    for j in range(STEPS):
        for k in range(STEPS):
            num_list_3d.append(loss_data_fin_3d_original[i][j][k])
plt.bar(range(len(num_list_3d)), num_list_3d)
plt.yticks(np.arange(2.0, 2.5, step=0.01))
plt.ylim(ymin=2.0, ymax=2.5)
plt.title('3D Loss Values for ' + args.model + ' model with original dataset')
plt.savefig('mnist_results/original/loss_mnist_3d_values_'+args.model+'.png')
plt.show()

plt.plot(range(len(num_list_3d)), num_list_3d,'o-',color = 'r',label="loss")
plt.title('3D Loss Values for ' + args.model + ' model with original dataset')
plt.savefig('mnist_results/original/loss_mnist_3d_values_line_'+args.model+'.png')
plt.show()

# plot loss contour in 2D
fig = plt.figure()
ax = plt.axes(projection='3d')
X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
ax.plot_surface(X, Y, loss_data_fin_original, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

cset = ax.contourf(X, Y, loss_data_fin_original, zdir='z', offset=3, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, loss_data_fin_original, zdir='x', offset=40, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, loss_data_fin_original, zdir='y', offset=0, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(0, 40)
ax.set_ylabel('Y')
ax.set_ylim(0, 40)
ax.set_zlabel('loss_data_fin')
ax.set_zlim(2, 3)

ax.set_title('Surface and Contour Plot of Loss Landscape for ' + args.model + ' model with original dataset')

# save plot to file and show
plt.savefig('mnist_results/original/loss_mnist_2d_plot_'+args.model+'.png')
plt.show()

# plot loss values of 2d loss landscape
num_list_2d = []
for i in range(STEPS):
    for j in range(STEPS):
            num_list_2d.append(loss_data_fin_original[i][j])
plt.bar(range(len(num_list_2d)), num_list_2d)
plt.yticks(np.arange(2.0, 2.5, step=0.01))
plt.ylim(ymin=2.0, ymax=2.5)
plt.title('2D Loss Values for ' + args.model + ' model with original dataset')
plt.savefig('mnist_results/original/loss_mnist_2d_values_'+args.model+'.png')
plt.show()

plt.plot(range(len(num_list_2d)), num_list_2d,'o-',color = 'r',label="loss")
plt.title('2D Loss Values for ' + args.model + ' model with original dataset')
plt.savefig('mnist_results/original/loss_mnist_2d_values_line_'+args.model+'.png')
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

# plot loss values of 3d loss landscape
num_list_c = []
for i in range(STEPS):
    for j in range(STEPS):
        for k in range(STEPS):
            num_list_c.append(loss_data_fin_3d_c[i][j][k])
plt.bar(range(len(num_list_c)), num_list_c)
plt.yticks(np.arange(2.0, 2.5, step=0.01))
plt.ylim(ymin=2.0, ymax=2.5)
plt.title('3D Loss Values for ' + args.model + ' model with ' + args.dataset + ' dataset')
plt.savefig(path+'loss_mnistc_3d_values_'+args.model+'.png')
plt.show()

plt.plot(range(len(num_list_c)), num_list_c,'o-',color = 'r',label="loss")
plt.title('3D Loss Values for ' + args.model + ' model with ' + args.dataset + ' dataset')
plt.savefig(path+'loss_mnistc_3d_values_line_'+args.model+'.png')
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
ax.set_title('Surface and Contour Plot of Loss Landscape for ' + args.model + ' model with ' + args.dataset + ' dataset')

# save plot to file and show
plt.savefig(path+'loss_mnistc_2d_plot_'+args.model+'.png')
plt.show()

# plot loss values of 2d loss landscape
num_list_2d_c = []
for i in range(STEPS):
    for j in range(STEPS):
            num_list_2d_c.append(loss_data_fin_c[i][j])
plt.bar(range(len(num_list_2d_c)), num_list_2d_c)
plt.yticks(np.arange(2.0, 2.5, step=0.01))
plt.ylim(ymin=2.0, ymax=2.5)
plt.title('2D Loss Values for ' + args.model + ' model with ' + args.dataset + ' dataset')
plt.savefig(path+'loss_mnistc_2d_values_'+args.model+'.png')
plt.show()

plt.plot(range(len(num_list_2d_c)), num_list_2d_c,'o-',color = 'r',label="loss")
plt.title('2D Loss Values for ' + args.model + ' model with ' + args.dataset + ' dataset')
plt.savefig(path+'loss_mnistc_2d_values_line_'+args.model+'.png')
plt.show()