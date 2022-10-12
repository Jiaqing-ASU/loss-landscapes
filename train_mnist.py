# libraries
import copy
import matplotlib
import numpy as np
from numpy import load
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse

matplotlib.rcParams['figure.figsize'] = [18, 12]

# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics

parser = argparse.ArgumentParser()
# input corruption dataset:
# original, brightness, canny_edges, dotted_line, fog, glass_blur, identity
# impulse_noise, motion_blur, rotate, scale, shear, shot_noise
# spatter, stripe, translate, zigzag
parser.add_argument('--dataset', default='original', help='training dataset')             
args = parser.parse_args()

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

if args.dataset == 'original':
    mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True, transform=Flatten())
elif args.dataset == 'brightness':
    x_c = load('mnist_c/brightness/train_images.npy')
    y_c = load('mnist_c/brightness/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'canny_edges':
    x_c = load('mnist_c/canny_edges/train_images.npy')
    y_c = load('mnist_c/canny_edges/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'dotted_line':
    x_c = load('mnist_c/dotted_line/train_images.npy')
    y_c = load('mnist_c/dotted_line/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'fog':
    x_c = load('mnist_c/fog/train_images.npy')
    y_c = load('mnist_c/fog/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'glass_blur':
    x_c = load('mnist_c/glass_blur/train_images.npy')
    y_c = load('mnist_c/glass_blur/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'identity':
    x_c = load('mnist_c/identity/train_images.npy')
    y_c = load('mnist_c/identity/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'impulse_noise':
    x_c = load('mnist_c/impulse_noise/train_images.npy')
    y_c = load('mnist_c/impulse_noise/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'motion_blur':
    x_c = load('mnist_c/motion_blur/train_images.npy')
    y_c = load('mnist_c/motion_blur/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'rotate':
    x_c = load('mnist_c/rotate/train_images.npy')
    y_c = load('mnist_c/rotate/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'scale':
    x_c = load('mnist_c/scale/train_images.npy')
    y_c = load('mnist_c/scale/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'shear':
    x_c = load('mnist_c/shear/train_images.npy')
    y_c = load('mnist_c/shear/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'shot_noise':
    x_c = load('mnist_c/shot_noise/train_images.npy')
    y_c = load('mnist_c/shot_noise/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'spatter':
    x_c = load('mnist_c/spatter/train_images.npy')
    y_c = load('mnist_c/spatter/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'stripe':
    x_c = load('mnist_c/stripe/train_images.npy')
    y_c = load('mnist_c/stripe/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'translate':
    x_c = load('mnist_c/translate/train_images.npy')
    y_c = load('mnist_c/translate/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])
elif args.dataset == 'zigzag':
    x_c = load('mnist_c/zigzag/train_images.npy')
    y_c = load('mnist_c/zigzag/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    mnist_train = []
    for i in range(len(x_c)):
        mnist_train.append([x_c[i], y_c[i]])

# define training data loader
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

path = 'mnist_model' + '/' + args.dataset + '/'
torch.save(model_initial.state_dict(), path + 'model_initial.pt')
torch.save(model_final.state_dict(), path + 'model_final.pt')