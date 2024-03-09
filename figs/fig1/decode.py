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
from sklearn.metrics import r2_score
import os
from bayes_opt import BayesianOptimization

stim = np.load('rawData/stim.npy')

numseeds = int(sys.argv[1])
Nh = int(sys.argv[2])

if not os.path.exists('decodeData'):
    os.mkdir('decodeData')
    
    
if not os.path.exists(f'decodeData/Nh{Nh}'):
    os.mkdir(f'decodeData/Nh{Nh}')

delta_t_list = [1,2,5,10,20,30,40,50,60,70,80,90,100]
numDeltas = len(delta_t_list)
layers = ['in','h','out']
numLayers = len(layers)
pconvec = [0.3]
numPcon = len(pconvec)

for pi in range(numPcon):
    pcon = pconvec[pi]
    print(f'pcon = {pcon}')
    if not os.path.exists(f'decodeData/Nh{Nh}/pcon{pcon}'):
        os.mkdir(f'decodeData/Nh{Nh}/pcon{pcon}')
    
    for seed in range(numseeds):
        print(f'\tseed {seed}')
        if not os.path.exists(f'decodeData/Nh{Nh}/pcon{pcon}/seed{seed}'):
            os.mkdir(f'decodeData/Nh{Nh}/pcon{pcon}/seed{seed}')
            
        for dti in range(numDeltas):
            deltat = delta_t_list[dti]
            
            for li in range(numLayers):
                layer = layers[li]
                
                loadpath = f'rawData/Nh{Nh}/pcon{pcon}/seed{seed}/{layer}_spks.npy'
                spk = np.load(loadpath)
            
                savepath = f'decodeData/Nh{Nh}/pcon{pcon}/seed{seed}/deltat{deltat}'
                if not os.path.exists(savepath):
                    os.mkdir(savepath)
        
                binnedTrains = getSlidingVecNew(spk,deltat)
                num_times = binnedTrains.shape[1]
                stimSub = stim[:num_times]
                y = stimSub
                X = binnedTrains.T
                X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.5,shuffle=False)
                X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, 
                                                                          test_size=0.5,shuffle=False)
                
                if li == 0: # save ytest only once per deltat
                    savepath_ytest = savepath + '/ytest.npy'
                    np.save(savepath_ytest,y_test)
                
                def svr(c):
                    svr_rbf_multi = MultiOutputRegressor(SVR(kernel="rbf", C=c))
                    svr_rbf_multi.fit(X_train,y_train)
                    r2_train = svr_rbf_multi.score(X_train,y_train)
                    r2_valid = svr_rbf_multi.score(X_validate,y_validate)
                    return r2_valid
                pbounds = {'c': (0.5, 10)}
                optimizer = BayesianOptimization(
                    f=svr,
                    pbounds=pbounds,
                    verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                    random_state=1,
                )
                optimizer.maximize(init_points=3, n_iter=1)
                bestC = optimizer.max['params']['c']
                svr_rbf_multi = MultiOutputRegressor(SVR(kernel="rbf", C=bestC))
                svr_rbf_multi.fit(X_train,y_train)
                y_pred = svr_rbf_multi.predict(X_test)
                    
                savepath_ypred = savepath + f'/ypred_{layer}.npy'
                np.save(savepath_ypred,y_pred)
    
