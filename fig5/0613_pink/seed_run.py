from help_funcs import *
# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

from scipy.fft import rfft, rfftfreq, irfft

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import itertools
import random
import statistics
import tqdm
import sys

from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import os
from bayes_opt import BayesianOptimization

if not os.path.exists('rawData'):
    os.mkdir('rawData')
# Nh_list = [int(x) for x in np.logspace(1,3,3)]
Nh = int(sys.argv[1])
stimseed = int(sys.argv[2])
modelseed = int(sys.argv[3])



alpha = 0.5


    

num_steps = 1000
tf = num_steps
dt = 1
tvec = np.arange(0,tf,dt)
flag = torch.cuda.is_available()
device = torch.device("cuda") if flag else torch.device("cpu")
if flag:
    print('device set to cuda')
    torch.set_default_device('cuda')
tf = num_steps
######################################################################
num_samples = 1
mode = "shot" # type of stimulus
batch_size = 1 # only one sample to learn
######################################################################

Nin = 100
Nout = 100

if not os.path.exists(f'rawData/Nh{Nh}'):
    os.mkdir(f'rawData/Nh{Nh}')
Ns = [Nin, Nh, Nout]
tag = 'post'

pcon=0.3
    
if not os.path.exists(f'rawData/Nh{Nh}'):
    os.mkdir(f'rawData/Nh{Nh}')

torch.manual_seed(stimseed)

if not os.path.exists(f'rawData/Nh{Nh}/stimseed{stimseed}'):
    os.mkdir(f'rawData/Nh{Nh}/stimseed{stimseed}')

dataset = RegressionDataset(timesteps=num_steps, num_samples=num_samples, mode=mode)

torch.manual_seed(modelseed)

if not os.path.exists(f'rawData/Nh{Nh}/stimseed{stimseed}/modelseed{modelseed}'):
    os.mkdir(f'rawData/Nh{Nh}/stimseed{stimseed}/modelseed{modelseed}')

modelName = f'trainedModels/Nh{Nh}/modelseed{modelseed}_stimseed{stimseed}.pth'
model = ConDivNet2dStimSparse_countTimeReadout(timesteps=num_steps, Nin=Nin, Nh=Nh, Nout=Nout, pcon=pcon, alpha=alpha).to(device)
model.load_state_dict(torch.load(modelName, map_location = device))


with torch.no_grad():
    feature = dataset.features
    label = dataset.labels
    feature = feature.to(device)
    s = label.to(device)
    readout, mem, spk1, spk2, spk3, spk4, spkh, spkout = model(s)


makeRaster2D(s.cpu(), spk1.cpu(), spk2.cpu(), spk3.cpu(), spk4.cpu(), spkh.cpu(), spkout.cpu(), readout.cpu(), Nh, stimseed, tag)

# if stimseed == 0:
stim = np.array(label[:,:,0].cpu().detach().numpy())
np.save(f'rawData/Nh{Nh}/stimseed{stimseed}/stim.npy',stim)

spk_in_1 = spk1.cpu().detach().numpy()
spk_in_2 = spk2.cpu().detach().numpy()
spk_in_3 = spk3.cpu().detach().numpy()
spk_in_4 = spk4.cpu().detach().numpy()
spk_h_rec = spkh.cpu().detach().numpy()
spk_out_rec = spkout.cpu().detach().numpy()
np.save(f'rawData/Nh{Nh}/stimseed{stimseed}/modelseed{modelseed}/readout.npy',readout.cpu().detach().numpy())


in_spks = np.hstack((spk_in_1,spk_in_2,spk_in_3,spk_in_4))
np.save(f'rawData/Nh{Nh}/stimseed{stimseed}/modelseed{modelseed}/in_spks.npy',in_spks)
np.save(f'rawData/Nh{Nh}/stimseed{stimseed}/modelseed{modelseed}/h_spks.npy',spk_h_rec)
np.save(f'rawData/Nh{Nh}/stimseed{stimseed}/modelseed{modelseed}/out_spks.npy',spk_out_rec)



