# libraries
from cgitb import lookup
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
from matplotlib import cm

# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics
import loss_landscapes.model_interface

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

def train(model, optimizer, criterion, train_loader, epochs):
    """ Trains the given model with the given optimizer, loss function, etc. """
    model.train()
    # train model
    for _ in tqdm(range(epochs), 'Training'):
        for count, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            x, y = batch

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    model.eval()

# download MNIST and setup data loaders
mnist_train = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)

# define model
model = MLPSmall(IN_DIM, OUT_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# stores the initial point in parameter space
model_initial = copy.deepcopy(model)

# train model
train(model, optimizer, criterion, train_loader, EPOCHS)
model_final = copy.deepcopy(model)

# data that the evaluator will use when evaluating loss
x, y = iter(train_loader).__next__()
metric = loss_landscapes.metrics.Loss(criterion, x, y)

# compute loss data
loss_data = loss_landscapes.linear_interpolation(model_initial, model_final, metric, STEPS, deepcopy_model=True)

# compute loss landscape 3d data
loss_data_fin = loss_landscapes.random_space(model_final, metric, 10, STEPS, normalization='filter', deepcopy_model=True)
print('loss_data_fin shape:', loss_data_fin.shape)

# reshape loss data
loss_data_fin.reshape(-1)

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
ax.scatter(X, Y, Z, c=loss_data_fin, cmap='rainbow')

# add plot labels
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'black'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'black'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'black'})

# save plot to file and show
plt.savefig('loss_mnist_3d.jpg')
plt.show()