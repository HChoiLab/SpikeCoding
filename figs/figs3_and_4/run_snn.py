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

numseeds = int(sys.argv[1])

num_steps = 1000
tf = num_steps
dt = 1
tvec = np.arange(0,tf,dt)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tf = num_steps
######################################################################
num_samples = 1
mode = "2dsos" # type of stimulus
batch_size = 1 # only one sample to learn
######################################################################

Nin = 48
NE1 = 1000
NB = 10
NE2 = 100
Nout = 10
Ns = [Nin, NE1, NB, NE2, Nout]
tag = 'post'

for seednum in range(numseeds):
    
    if not os.path.exists(f'rawData/seed{seednum}'):
        os.mkdir(f'rawData/seed{seednum}')
    
    print(f'seed {seednum}')
    torch.manual_seed(seednum)
    random.seed(seednum)
    np.random.seed(seednum)
    # modelName = f'trainedModels/model_Nh{Nh}_seed{seednum}.pth'
    dataset = RegressionDataset_getFreqs(timesteps=num_steps,num_samples=num_samples, mode=mode)
    modelName = f'trainedModels/seed{seednum}.pth'
    model = ConDivNet2dStimSparse5layer(timesteps=num_steps, Nin=Nin, NE1=NE1, NB=NB, NE2=NE2, Nout=Nout).to(device)
    model.load_state_dict(torch.load(modelName))
   

    with torch.no_grad():
            feature = dataset.features
            label = dataset.labels
            label4Hz = dataset.labels1
            label20Hz = dataset.labels2
            feature = feature.to(device)
            s = label.to(device)
            mem, spk1, spk2, spk3, spk4, spkE1, spkB, spkE2, spkout = model(s)
            
    plot2DfitAlt(s,mem,seednum,tag)
    makeRaster2D_5layer(s, spk1, spk2, spk3, spk4, spkE1, spkB, spkE2, spkout, mem, seednum, tag)
    
    if seednum == 0:
        stim = np.array(label[:,:,0])
        np.save(f'rawData/stim.npy',stim)
        stim4Hz = np.array(label4Hz[:,:,0])
        np.save(f'rawData/stim4Hz.npy',stim4Hz)
        stim20Hz = np.array(label20Hz[:,:,0])
        np.save(f'rawData/stim20Hz.npy',stim20Hz)
        
    mem_readout = np.array(mem[:,:,0])
    np.save(f'rawData/seed{seednum}/mem.npy',mem_readout)
    spk_in_1 = spk1.detach().numpy()
    spk_in_2 = spk2.detach().numpy()
    spk_in_3 = spk3.detach().numpy()
    spk_in_4 = spk4.detach().numpy()
    spk_E1_rec = spkE1.detach().numpy()
    spk_B_rec = spkB.detach().numpy()
    spk_E2_rec = spkE2.detach().numpy()
    spk_out_rec = spkout.detach().numpy()

    spk_in_1_re = np.reshape(spk_in_1,(num_steps,int(Nin/4)))
    in_1_spks = np.array(spk_in_1_re,dtype=bool)
    st1, si1 = np.where(in_1_spks == True)

    spk_in_2_re = np.reshape(spk_in_2,(num_steps,int(Nin/4)))
    in_2_spks = np.array(spk_in_2_re,dtype=bool)
    st2, si2 = np.where(in_2_spks == True)

    spk_in_3_re = np.reshape(spk_in_3,(num_steps,int(Nin/4)))
    in_3_spks = np.array(spk_in_3_re,dtype=bool)
    st3, si3 = np.where(in_3_spks == True)

    spk_in_4_re = np.reshape(spk_in_4,(num_steps,int(Nin/4)))
    in_4_spks = np.array(spk_in_4_re,dtype=bool)
    st4, si4 = np.where(in_4_spks == True)
    
    in_spks = np.hstack((spk_in_1,spk_in_2,spk_in_3,spk_in_4))
    np.save(f'rawData/seed{seednum}/in_spks.npy',in_spks)

    spk_E1_rec_re = np.reshape(spk_E1_rec,(num_steps,NE1))
    E1_spks = np.array(spk_E1_rec_re,dtype=bool)
    np.save(f'rawData/seed{seednum}/E1_spks.npy',E1_spks)

    spk_B_rec_re = np.reshape(spk_B_rec,(num_steps,NB))
    B_spks = np.array(spk_B_rec_re,dtype=bool)
    np.save(f'rawData/seed{seednum}/B_spks.npy',B_spks)

    spk_E2_rec_re = np.reshape(spk_E2_rec,(num_steps,NE2))
    E2_spks = np.array(spk_E2_rec_re,dtype=bool)
    np.save(f'rawData/seed{seednum}/E2_spks.npy',E2_spks)

    spk_out_rec_re = np.reshape(spk_out_rec,(num_steps,Nout))
    out_spks = np.array(spk_out_rec_re,dtype=bool)
    np.save(f'rawData/seed{seednum}/out_spks.npy',out_spks)
    


