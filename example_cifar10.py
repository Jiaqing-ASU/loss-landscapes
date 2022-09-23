# libraries
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from numpy import save
from matplotlib import cm

import argparse
import copy
import h5py
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import dataloader
import evaluation
import projection as proj
import net_plotter
import model_loader
import scheduler
import mpi4pytorch as mpi

# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    
    net = torchvision.models.resnet50(pretrained = False)
    net = net.cpu()
    net.fc = nn.Linear(2048, 10)
    net.load_state_dict(torch.load("resnet-model.pt",map_location=torch.device('cpu')))
    
    w = net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    net.eval()
    model_final = copy.deepcopy(net)
    STEPS = 40
    
    train_loader, test_loader = dataloader.load_dataset('cifar10','cifar10/data', 32, 1, False,1, 0,'','')
    x,target = iter(train_loader).__next__()
    target = target.unsqueeze(1)
    target_hot = torch.FloatTensor(torch.zeros((target.size()[0],10)).scatter(1,target,1.0))
    metric = loss_landscapes.metrics.Loss(torch.nn.CrossEntropyLoss(),x,target_hot)
    loss_data_fin = loss_landscapes.random_plane(net,metric,10,STEPS,normalization='filter',deepcopy_model=True)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    
    cset = ax.contourf(X, Y, loss_data_fin, zdir='z', offset=20, cmap=cm.coolwarm)
    #cset = ax.contourf(X, Y, loss_data_fin, zdir='x', offset=40, cmap=cm.coolwarm)
    #cset = ax.contourf(X, Y, loss_data_fin, zdir='y', offset=0, cmap=cm.coolwarm)
    
    ax.set_xlabel('X')
    ax.set_xlim(0, 40)
    ax.set_ylabel('Y')
    ax.set_ylim(0, 40)
    ax.set_zlabel('loss_data_fin')
    ax.set_zlim(-20, 20)
    
    ax.set_title('Surface and Contour Plot of Loss Landscape')
    
    # save plot to file and show
    plt.savefig('loss_cifar10_2d_plot.png')
    plt.show()


    # compute loss landscape 3D data
    loss_data_fin_3d = loss_landscapes.random_space(model_final, metric, 10, STEPS, normalization='filter', deepcopy_model=True)

    # reshape loss data for 3D plot
    loss_data_fin_3d.reshape(-1)

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
    ax.scatter(X, Y, Z, c=loss_data_fin_3d, cmap='rainbow')

    # add plot labels
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'black'})

    # save plot to file and show
    plt.savefig('loss_mnist_3d.png')
    plt.show()