from help_funcs import *
# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

# print('imported snntorch stuff')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# print('imported torch stuff')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import itertools
import random
import statistics
import tqdm
import sys

# print('imported matplotlib, numpy,...')

from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import os
from bayes_opt import BayesianOptimization


num_steps = 1000
tf = num_steps
dt = 1
tvec = np.arange(0,tf,dt)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tf = num_steps
######################################################################
num_samples = 1
mode = "2dsineDis" # type of stimulus
batch_size = 1 # only one sample to learn
######################################################################
Nin = 100
Nh = int(sys.argv[1])
pcon = float(sys.argv[2])
Nout = 100
Ns = [Nin, Nh, Nout]
layerNames = ['in', 'h', 'out']
numLayers = len(layerNames)
N_nrnVec = Ns
T_R = 50
delay = 0
delta_t = 2
train_frac = 0.5
slideVec = np.arange(0,tf-T_R+dt,dt)
tag = 'post'

numseeds = 10

for seednum in range(numseeds):
        print(f'seed {seednum}')
        torch.manual_seed(seednum)
        random.seed(seednum)
        np.random.seed(seednum)
        dataset = RegressionDataset(timesteps=num_steps, num_samples=num_samples, mode=mode)
        model = ConDivNet2dStimSparse(timesteps=num_steps, Nin=Nin, Nh=Nh, Nout=Nout, pcon=pcon).to(device)
        
        # locate zero-value weights before training loop
        EPS = 1e-6
        locked_masks = {n: torch.abs(w) < EPS for n, w in model.named_parameters() if n.endswith('weight')} 
        
        loss_function = torch.nn.MSELoss()
        num_iter = 100 # train for this many iterations
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        loss_hist = [] # record loss


        # training loop
        with tqdm.trange(num_iter) as pbar:
                for _ in pbar:
                        minibatch_counter = 0
                        loss_epoch = []
                        feature = dataset.features
                        label = dataset.labels
                        feature = feature.to(device)
                        s = label.to(device)
                        mem, spk1, spk2, spk3, spk4, spkh, spkout = model(s)
                        loss_val = loss_function(mem, s) # calculate loss
                        optimizer.zero_grad() # zero out gradients
                        loss_val.backward() # calculate gradients
                        
                        for n, w in model.named_parameters():
                            if w.grad is not None and n in locked_masks:
                                w.grad[locked_masks[n]] = 0
                        
                        optimizer.step() # update weights
                #         # store loss
                        loss_hist.append(loss_val.item())
                        minibatch_counter += 1
                        pbar.set_postfix(loss="%.3e" % loss_val.item()) # print loss p/batch
#         if not os.path.exists('lossPlots'):
#                 os.mkdir('lossPlots')
#         plt.plot(loss_hist,color='black')
#         plt.ylabel('mse loss',fontsize=18)
#         plt.xlabel('iteration no.',fontsize=18)
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=12)
#         fname = f'lossPlots/loss_v_iter_Nh{Nh}_seed{seednum}.png'
#         plt.savefig(fname,bbox_inches='tight',dpi=200)
#         plt.close()

#         plot2Dfit(s,mem,seednum,Nh,tag)
#         makeRaster2D(s, spk1, spk2, spk3, spk4, spkh, spkout, mem, Nh, seednum, tag)

        if not os.path.exists('trainedModels'):
                os.mkdir('trainedModels')
        if not os.path.exists(f'trainedModels/Nh{Nh}'):
                os.mkdir(f'trainedModels/Nh{Nh}')
        if not os.path.exists(f'trainedModels/Nh{Nh}/pcon{pcon}'):
                os.mkdir(f'trainedModels/Nh{Nh}/pcon{pcon}')
        fname = f'trainedModels/Nh{Nh}/pcon{pcon}/seed{seednum}.pth'
        torch.save(model.state_dict(), fname)
        
        
