# libraries
import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
import net_plotter
import dataloader
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
    
    counters = plt.contour(loss_data_fin,levels=10,colors='black')
    plt.clabel(counters,inline=True,fontsize=8)
    plt.imshow(loss_data_fin,extent=[0,STEPS,0,STEPS],origin='lower',cmap='viridis',alpha=0.5)
    plt.colorbar()
    plt.title('Contour Plot of Loss Landscape')
    plt.savefig('loss_net_contour.png')
    
    print("loss_data_fin")
    print(loss_data_fin)
    print(loss_data_fin.shape)