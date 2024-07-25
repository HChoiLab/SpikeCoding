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
import torchaudio

# print('imported torch stuff')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import itertools
import random
import statistics
import tqdm

# print('imported matplotlib, numpy,...')

from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import os
from bayes_opt import BayesianOptimization

# print('imported ML stuff')

def getStrong(spks,T):
    tf = spks.shape[0]
    tmax = tf-T+1
    Nnrn = spks.shape[1]
    binned_spks = np.zeros((tmax,T*Nnrn))
    for t in range(tmax):
        binned_spks[t,:] = spks[t:t+T,:].flatten()
    return binned_spks

def getStrong2(spks,T,deltat):
    kernel = np.ones(deltat)
    tf = spks.shape[0]
    tmax = tf-T+1
    Nnrn = spks.shape[1]
    binned_spks = np.zeros((tmax,(T-deltat+1)*Nnrn))
    for t in range(tmax):
        pre = np.array([np.convolve(spks[t:t+T,i],kernel,mode='valid') for i in range(Nnrn)])
        binned_spks[t,:] = pre.flatten()
    return binned_spks

def getStrong3(spks,T,deltat):
    kernel = np.ones(deltat)
    tf = spks.shape[0]
    tmax = tf-T+1
    Nnrn = spks.shape[1]
    # X=np.zeros((n_samples,n_time_bins,n_neurons))
    binned_spks = np.zeros((tmax,(T-deltat+1),Nnrn))
    for t in range(tmax):
        for nrn in range(Nnrn):
            binned_spks[t,:,nrn] = np.convolve(spks[t:t+T,nrn],kernel,mode='valid')
    return binned_spks

def getSlidingVecNew(spk,delta_t,kernelType='gauss'):
    if delta_t < 1:
        print('delta_t must be an integer greater than or equal to 1')
        countVec = 0
    else:
        if kernelType == 'gauss':
            kernelRange = np.linspace(0,1,delta_t+1)
            half = kernelRange[int(delta_t/2)]
            sigma = 0.1
            kernel = [np.exp(-((x-half)**2)/sigma) for x in kernelRange]
        elif kernelType == 'rect':
            kernel = np.ones(delta_t+1)
        countVec = np.array([np.convolve(spk[:,i],kernel,mode='valid') for i in range(spk.shape[1])])
    return countVec


def getRank(st,delta_t,t1,T_R):
    st_mod = st-t1
    n_time_bins = int(T_R/delta_t)
    binTrain = np.zeros(n_time_bins,dtype='int')
    timeVec = np.arange(0,T_R+delta_t,delta_t)
    counter = 0
    while counter < len(timeVec)-1:
        timeRange0 = counter*delta_t
        sub_st = st_mod[st_mod > timeRange0]
        timeRange1 = (counter+1)*delta_t
        sub_st = sub_st[sub_st < timeRange1]
        if len(sub_st) == 1:
            binTrain[counter] = int(1)
        elif len(sub_st) > 1:
            print(f'more than one spike in a single time bin of size delta_t = {delta_t}')
        counter += 1
    rankStr = ''
    for x in binTrain:
        rankStr += str(x)
    rank = int(rankStr,2)
    
    return rank

def getRankLog(st,delta_t,t1,T_R):
    st_mod = st-t1
    n_time_bins = int(T_R/delta_t)
    binTrain = np.zeros(n_time_bins,dtype='int')
    timeVec = np.arange(0,T_R+delta_t,delta_t)
    counter = 0
    while counter < len(timeVec)-1:
        timeRange0 = counter*delta_t
        sub_st = st_mod[st_mod > timeRange0]
        timeRange1 = (counter+1)*delta_t
        sub_st = sub_st[sub_st < timeRange1]
        if len(sub_st) == 1:
            binTrain[counter] = int(1)
        elif len(sub_st) > 1:
            print(f'more than one spike in a single time bin of size delta_t = {delta_t}')
        counter += 1
    rankStr = ''
    for x in binTrain:
        rankStr += str(x)
    rank = int(rankStr,2)
    if rank > 0:
        rank = np.log2(rank)
    return rank


def getSlidingCountRankVecs(spikeTimes,spikeIndices,N_nrn,tf,T_R,dt,delta_t,seednum,delay):
    slideVec = np.arange(delay,tf-T_R+dt,dt)
    countVec = np.zeros((N_nrn,len(slideVec)))
    rankVec = np.zeros((N_nrn,len(slideVec)))
    for ni in range(N_nrn):
        st = spikeTimes[spikeIndices==ni]
        ti = 0 # time index
        for t in slideVec:
            t1 = t
            t2 = t+T_R
            sub_st = st[st > t1]
            sub_st = sub_st[sub_st < t2] - t1
            countVec[ni][ti] = len(sub_st)
            rankVec[ni][ti] = getRank(st,delta_t,t1,T_R)
            ti+=1
    
    return [countVec, rankVec]

def getSlidingVecs(spikeTimes,spikeIndices,N_nrn,tf,T_R,dt,delta_t,seednum,delay):
    slideVec = np.arange(delay,tf-T_R+dt,dt)
    maxSpikes = int(T_R/delta_t)
    timeVec = np.zeros((N_nrn*maxSpikes,len(slideVec)))
    countVec = np.zeros((N_nrn,len(slideVec)))
    rankVec = np.zeros((N_nrn,len(slideVec)))
    tn = 0
    for ni in range(N_nrn):
        st = spikeTimes[spikeIndices==ni]
        ti = 0 # time index
        for t in slideVec:
            t1 = t
            t2 = t+T_R
            sub_st = st[st > t1]
            sub_st = sub_st[sub_st < t2] - t1
            count = len(sub_st)
            timeVec[tn:tn+count,ti] = sub_st
            countVec[ni][ti] = count
            rankVec[ni][ti] = getRank(st,delta_t,t1,T_R)
            ti+=1
        tn += maxSpikes
    
    return [timeVec, countVec, rankVec]

def getSlidingVecsLog(spikeTimes,spikeIndices,N_nrn,tf,T_R,dt,delta_t,seednum,delay):
    slideVec = np.arange(delay,tf-T_R+dt,dt)
    maxSpikes = int(T_R/delta_t)
    timeVec = np.zeros((N_nrn*maxSpikes,len(slideVec)))
    # timeVec[:] = np.nan
    countVec = np.zeros((N_nrn,len(slideVec)))
    rankVec = np.zeros((N_nrn,len(slideVec)))
    tn = 0
    for ni in range(N_nrn):
        st = spikeTimes[spikeIndices==ni]
        ti = 0 # time index
        for t in slideVec:
            t1 = t
            t2 = t+T_R
            sub_st = st[st > t1]
            sub_st = sub_st[sub_st < t2] - t1
            count = len(sub_st)
            timeVec[tn:tn+count,ti] = sub_st
            countVec[ni][ti] = count
            rankVec[ni][ti] = getRankLog(st,delta_t,t1,T_R)
            ti+=1
        tn += maxSpikes
    
    return [timeVec, countVec, rankVec]


class RegressionDataset(torch.utils.data.Dataset):
    """Simple regression dataset."""

    def __init__(self, timesteps, num_samples, mode):
        """Linear relation between input and output"""
        self.num_samples = num_samples # number of generated samples
        feature_lst = [] # store each generated sample in a list
        
        f1 = 2.5
        f2 = 5.
        f3 = 10.
        amp = 0.4

        # generate linear functions one by one
        for idx in range(num_samples):
#             end = float(torch.rand(1)) # random final point
            end = float(1)
            lin_vec = torch.linspace(start=0.0, end=end, steps=timesteps) # generate linear function from 0 to end
            feature = lin_vec.view(timesteps, 1)
            # feature = lin_vec.view(timesteps)
            feature_lst.append(feature) # add sample to list

        # self.features = torch.stack(feature_lst) # convert list to tensor
        self.features = torch.stack(feature_lst, dim=1) # convert list to tensor
        # print(f'self.features = {self.features}')

        # option to generate linear function or square-root function
        if mode == "linear":
            self.labels = self.features * 1

        elif mode == "sqrt":
            slope = float(torch.rand(1))
            self.labels = torch.sqrt(self.features * slope)
            
        elif mode == 'sine':
#             print(self.features)
            self.labels = 0.9*torch.sin(2.*torch.pi*5.*self.features)
    
        elif mode == '2dsine':
#             print(self.features)
            s = 0.3*torch.sin(2.*torch.pi*5.*self.features)
            c = 0.9*torch.cos(2.*torch.pi*5.*self.features)
            self.labels = torch.hstack((c,s))
            # print(self.labels.shape)

        elif mode == '2dsineDis':
            ells = 0.3*torch.sin(2.*torch.pi*5.*self.features)
            ellc = 0.9*torch.cos(2.*torch.pi*5.*self.features)
            s = ells*(self.features < 0.25) + ellc*(self.features > 0.25)*(self.features < 0.85) + ells*(self.features > 0.85)
            c = ellc*(self.features < 0.25) + ells*(self.features > 0.25)*(self.features < 0.85) + ellc*(self.features > 0.85)
            self.labels = torch.hstack((c,s))

        elif mode == '2dsos':
            a1 = 0.4
            a2 = 0.4
            f1 = 4.
            f2 = 20.
            c1 = a1*torch.cos(2.*torch.pi*f1*self.features)
            c2 = a2*torch.cos(2.*torch.pi*f2*self.features)
            c = c1 + c2
            s1 = a1*torch.sin(2.*torch.pi*f1*self.features)
            s2 = a2*torch.sin(2.*torch.pi*f2*self.features)
            s = s1 + s2
            self.labels = torch.hstack((c,s))
            
    
        elif mode == 'sos':
            self.labels = amp*torch.sin(2.*torch.pi*f1*self.features) + amp*torch.sin(2.*torch.pi*f2*self.features)

        else:
            raise NotImplementedError("'linear', 'sqrt', 'sine'")

    def __len__(self):
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """General implementation, but we only have one sample."""
        
        return self.features[:, idx, :], self.labels[:, :2, 0]
    
    
class RegressionDataset_wFreqs(torch.utils.data.Dataset):
    """Simple regression dataset."""

    def __init__(self, timesteps, num_samples, mode):
        """Linear relation between input and output"""
        self.num_samples = num_samples # number of generated samples
        feature_lst = [] # store each generated sample in a list
        
        f1 = 2.5
        f2 = 5.
        f3 = 10.
        amp = 0.4

        # generate linear functions one by one
        for idx in range(num_samples):
#             end = float(torch.rand(1)) # random final point
            end = float(1)
            lin_vec = torch.linspace(start=0.0, end=end, steps=timesteps) # generate linear function from 0 to end
            feature = lin_vec.view(timesteps, 1)
            # feature = lin_vec.view(timesteps)
            feature_lst.append(feature) # add sample to list

        # self.features = torch.stack(feature_lst) # convert list to tensor
        self.features = torch.stack(feature_lst, dim=1) # convert list to tensor
        # print(f'self.features = {self.features}')

        # option to generate linear function or square-root function
        if mode == "linear":
            self.labels = self.features * 1

        elif mode == "sqrt":
            slope = float(torch.rand(1))
            self.labels = torch.sqrt(self.features * slope)
            
        elif mode == 'sine':
#             print(self.features)
            self.labels = 0.9*torch.sin(2.*torch.pi*5.*self.features)
    
        elif mode == '2dsine':
#             print(self.features)
            s = 0.3*torch.sin(2.*torch.pi*5.*self.features)
            c = 0.9*torch.cos(2.*torch.pi*5.*self.features)
            self.labels = torch.hstack((c,s))
            # print(self.labels.shape)

        elif mode == '2dsineDis':
            ells = 0.3*torch.sin(2.*torch.pi*5.*self.features)
            ellc = 0.9*torch.cos(2.*torch.pi*5.*self.features)
            s = ells*(self.features < 0.25) + ellc*(self.features > 0.25)*(self.features < 0.85) + ells*(self.features > 0.85)
            c = ellc*(self.features < 0.25) + ells*(self.features > 0.25)*(self.features < 0.85) + ellc*(self.features > 0.85)
            self.labels = torch.hstack((c,s))

        elif mode == '2dsos':
            a1 = 0.4
            a2 = 0.4
            f1 = 4.
            f2 = 20.
            c1 = a1*torch.cos(2.*torch.pi*f1*self.features)
            c2 = a2*torch.cos(2.*torch.pi*f2*self.features)
            c = c1 + c2
            s1 = a1*torch.sin(2.*torch.pi*f1*self.features)
            s2 = a2*torch.sin(2.*torch.pi*f2*self.features)
            s = s1 + s2
            self.labels1 = torch.hstack((c1,s1))
            self.labels2 = torch.hstack((c2,s2))
            self.labels = torch.hstack((c,s))
            
    
        elif mode == 'sos':
            self.labels = amp*torch.sin(2.*torch.pi*f1*self.features) + amp*torch.sin(2.*torch.pi*f2*self.features)

        else:
            raise NotImplementedError("'linear', 'sqrt', 'sine'")

    def __len__(self):
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """General implementation, but we only have one sample."""
        
        return self.features[:, idx, :], self.labels[:, :2, 0], self.labels1[:, :2, 0], self.labels2[:, :2, 0]
    
    
class RegressionDataset2D(torch.utils.data.Dataset):
    """Simple regression dataset."""

    def __init__(self, timesteps, num_samples, mode):
        """Linear relation between input and output"""
        self.num_samples = num_samples # number of generated samples
        feature_lst = [] # store each generated sample in a list
        
        f1 = 2.5
        f2 = 5.
        f3 = 10.
        amp = 0.4

        # generate linear functions one by one
        for idx in range(num_samples):
#             end = float(torch.rand(1)) # random final point
            end = float(1)
            lin_vec = torch.linspace(start=0.0, end=end, steps=timesteps) # generate linear function from 0 to end
            feature = lin_vec.view(timesteps, 1)
            feature_lst.append(feature) # add sample to list

#         self.features = torch.stack(feature_lst, dim=1) # convert list to tensor
        feat_lst = torch.tensor(feature_lst)
        self.features = torch.cat((feat_lst,feat_lst), dim=1) # convert list to tensor
        # print(f'self.features.shape = {self.features.shape}')

        # option to generate linear function or square-root function
        if mode == "linear":
            self.labels = self.features * 1

        elif mode == "sqrt":
            slope = float(torch.rand(1))
            self.labels = torch.sqrt(self.features * slope)
            
        elif mode == 'sine':
#             print(self.features)
            self.labels = 0.9*torch.sin(2.*torch.pi*5.*self.features)
    
        elif mode == '2dsine':
#             print(self.features)
            s = 0.3*torch.sin(2.*torch.pi*5.*self.features)
            c = 0.9*torch.cos(2.*torch.pi*5.*self.features)
            self.labels = torch.hstack((c,s))
    
        elif mode == 'sos':
            self.labels = amp*torch.sin(2.*torch.pi*f1*self.features) + amp*torch.sin(2.*torch.pi*f2*self.features)

        else:
            raise NotImplementedError("'linear', 'sqrt', 'sine'")

    def __len__(self):
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """General implementation, but we only have one sample."""
        
        return self.features[:, idx, :], self.labels[:, :2, :]
    
class stim_freq(torch.utils.data.Dataset):
    """Simple regression dataset."""

    def __init__(self, timesteps, num_samples, f):
        """Linear relation between input and output"""
        self.num_samples = num_samples # number of generated samples
        feature_lst = [] # store each generated sample in a list
       

        # generate linear functions one by one
        for idx in range(num_samples):
            end = float(1)
            lin_vec = torch.linspace(start=0.0, end=end, steps=timesteps) # generate linear function from 0 to end
            feature = lin_vec.view(timesteps, 1)
            feature_lst.append(feature) # add sample to list

        self.features = torch.stack(feature_lst, dim=1) # convert list to tensor


        a1 = 0.4
        a2 = 0.4
        f1 = 4.
        f2 = f # higher variable frequency
        c1 = a1*torch.cos(2.*torch.pi*f1*self.features)
        c2 = a2*torch.cos(2.*torch.pi*f2*self.features)
        c = c1 + c2
        s1 = a1*torch.sin(2.*torch.pi*f1*self.features)
        s2 = a2*torch.sin(2.*torch.pi*f2*self.features)
        s = s1 + s2
        self.labels1 = torch.hstack((c1,s1))
        self.labels2 = torch.hstack((c2,s2))
        self.labels = torch.hstack((c,s))
        
    def __len__(self):
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """General implementation, but we only have one sample."""
        
        return self.features[:, idx, :], self.labels[:, :2, 0], self.labels1[:,:2,0], self.labels2[:,:2,0]
    
class ConDivNet2dStimSparse_readout(torch.nn.Module):
    """spiking neural network in snntorch."""

    def __init__(self, timesteps, Nin, Nh, Nout, pcon):
        super().__init__()
        
        self.timesteps = timesteps # number of time steps to simulate the network
        self.inp_1 = int(Nin/4)
        self.inp_2 = int(Nin/4)
        self.inp_3 = int(Nin/4)
        self.inp_4 = int(Nin/4)
        self.h = Nh # number of hidden neurons 
        self.out = Nout
        self.p = pcon # connection sparsity (% of connections to set to zero)
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function
        self.inp_1_offsets = 0.1*torch.randn(self.inp_1)
        self.inp_2_offsets = 0.1*torch.randn(self.inp_2)
        self.inp_3_offsets = 0.1*torch.randn(self.inp_3)
        self.inp_4_offsets = 0.1*torch.randn(self.inp_4)
        
        self.readout_vec = torch.nn.Parameter(0.1*torch.randn(Nout,2))
#         print(f'self.readout_vec.shape = {self.readout_vec.shape}')
        
        # global parameters
        alpha_min = 0.7
        alpha_max = 0.9
        thr_min = 0.0
        thr_max = 0.5
        
##################################################################################################
        
        # stimulus to quadrant 1 input layer
        self.fc_in_1 = torch.nn.Linear(in_features=self.inp_1, out_features=self.inp_1, bias=False)
#         torch.nn.init.sparse_(self.fc_in_1.weight, sparsity=self.p, std=1./np.sqrt(1*self.p))
        # randomly initialize parameters for input layer
        alpha_in_1 = (alpha_max-alpha_min)*torch.rand(self.inp_1) + alpha_min
        beta_in_1 = alpha_in_1 - 0.1
        thr_in_1 = torch.rand(self.inp_1)*(thr_max-thr_min) + thr_min
        self.lif_in_1 = snn.Alpha(alpha=alpha_in_1, beta=beta_in_1, 
                                threshold=thr_in_1, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 2 input layer
        self.fc_in_2 = torch.nn.Linear(in_features=self.inp_2, out_features=self.inp_2, bias=False)
#         torch.nn.init.sparse_(self.fc_in_2.weight, sparsity=self.p, std=1./np.sqrt(1*self.p))
        # randomly initialize parameters for input layer
        alpha_in_2 = (alpha_max-alpha_min)*torch.rand(self.inp_2) + alpha_min
        beta_in_2 = alpha_in_2 - 0.1
        thr_in_2 = torch.rand(self.inp_2)*(thr_max-thr_min) + thr_min
        self.lif_in_2 = snn.Alpha(alpha=alpha_in_2, beta=beta_in_2, 
                                threshold=thr_in_2, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 3 input layer
        self.fc_in_3 = torch.nn.Linear(in_features=self.inp_3, out_features=self.inp_3, bias=False)
#         torch.nn.init.sparse_(self.fc_in_3.weight, sparsity=self.p, std=1.)
        # randomly initialize parameters for input layer
        alpha_in_3 = (alpha_max-alpha_min)*torch.rand(self.inp_3) + alpha_min
        beta_in_3 = alpha_in_3 - 0.1
        thr_in_3 = torch.rand(self.inp_3)*(thr_max-thr_min) + thr_min
        self.lif_in_3 = snn.Alpha(alpha=alpha_in_3, beta=beta_in_3, 
                                threshold=thr_in_3, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 4 input layer
        self.fc_in_4 = torch.nn.Linear(in_features=self.inp_4, out_features=self.inp_4, bias=False)
#         torch.nn.init.sparse_(self.fc_in_4.weight, sparsity=self.p, std=1.)
        # randomly initialize parameters for input layer
        alpha_in_4 = (alpha_max-alpha_min)*torch.rand(self.inp_4) + alpha_min
        beta_in_4 = alpha_in_4 - 0.1
        thr_in_4 = torch.rand(self.inp_4)*(thr_max-thr_min) + thr_min
        self.lif_in_4 = snn.Alpha(alpha=alpha_in_4, beta=beta_in_4, 
                                threshold=thr_in_4, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        
        # quadrant 1 input layer to E1 layer
        self.fc_h_1 = torch.nn.Linear(in_features=self.inp_1, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_1.weight, sparsity=self.p, std=1./np.sqrt(self.inp_1*(1-self.p)))
        # quadrant 2 input layer to E1 layer
        self.fc_h_2 = torch.nn.Linear(in_features=self.inp_2, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_2.weight, sparsity=self.p, std=1./np.sqrt(self.inp_2*(1-self.p)))
        # quadrant 1 input layer to E1 layer
        self.fc_h_3 = torch.nn.Linear(in_features=self.inp_3, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_3.weight, sparsity=self.p, std=1./np.sqrt(self.inp_3*(1-self.p)))
        # quadrant 2 input layer to E1 layer
        self.fc_h_4 = torch.nn.Linear(in_features=self.inp_4, out_features=self.h, bias=False) 
        torch.nn.init.sparse_(self.fc_h_4.weight, sparsity=self.p, std=1./np.sqrt(self.inp_4*(1-self.p)))
        # randomly initialize parameters for E1 layer
        alpha_h = (alpha_max-alpha_min)*torch.rand(self.h) + alpha_min
        beta_h = alpha_h - 0.1
        thr_h = torch.rand(self.h)*(thr_max-thr_min) + thr_min
        self.lif_h = snn.Alpha(alpha=alpha_h, beta=beta_h, threshold=thr_h, 
                                learn_threshold=False, spike_grad=spike_grad)
        
##################################################################################################

        # hidden layer to output layer
        self.fc_out = torch.nn.Linear(in_features=self.h, out_features=self.out,bias=False)
        torch.nn.init.sparse_(self.fc_out.weight, sparsity=self.p, std=1./np.sqrt(self.h*(1-self.p)))
        # randomly initialize parameters for B layer
        alpha_out = (alpha_max-alpha_min)*torch.rand(self.out) + alpha_min
        beta_out = alpha_out - 0.1
        thr_out = torch.rand(self.out)*(thr_max-thr_min) + thr_min
        self.lif_out = snn.Alpha(alpha=alpha_out, beta=beta_out, threshold=thr_out, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # output layer to readout neuron
        self.fc_readout = torch.nn.Linear(in_features=self.out, out_features=2,bias=False)        
        # randomly initialize decay rate for single output neuron
        thr_readout = torch.ones(2)
        beta_readout = torch.rand(2)
        # leaky integrator readout neuron used for reconstruction
        self.li_readout = snn.Leaky(beta=beta_readout, threshold=thr_readout, learn_beta=True, 
                                    spike_grad=spike_grad, reset_mechanism="none")
        

    def forward(self, stim):
        """Forward pass for several time steps."""

        # Initalize membrane potential
#         pre_in_pos, post_in_pos, mem_in_pos = self.lif_in_pos.init_alpha()
#         pre_in_neg, post_in_neg, mem_in_neg = self.lif_in_neg.init_alpha()
        pre_in_1, post_in_1, mem_in_1 = self.lif_in_1.init_alpha()
        pre_in_2, post_in_2, mem_in_2 = self.lif_in_2.init_alpha()
        pre_in_3, post_in_3, mem_in_3 = self.lif_in_3.init_alpha()
        pre_in_4, post_in_4, mem_in_4 = self.lif_in_4.init_alpha()
        presyn_h, postsyn_h, mem_h = self.lif_h.init_alpha()
        presyn_out, postsyn_out, mem_out = self.lif_out.init_alpha()
        mem_readout = self.li_readout.init_leaky()

        # Empty lists to record outputs
        spk_in_1_rec = []
        spk_in_2_rec = []
        spk_in_3_rec = []
        spk_in_4_rec = []
        spk_h_rec = []
        spk_out_rec = []
        mem_readout_rec = []
#         readout_rec = []
        
        deltat = 10
        kernelRange = torch.linspace(0,1,deltat+1)
        sigma = 0.1
        half = kernelRange[int(deltat/2)]
        k = torch.exp(-((kernelRange-half)**2)/sigma)
        kernel = torch.zeros((self.out,deltat+1))
        for i in range(self.out):
            kernel[i,:] = k
#         kernel = torch.tensor([torch.exp(-((x-half)**2)/sigma) for x in kernelRange])
#         plt.plot(kernel)
#         plt.show()
#         plt.close()
        
        # print(f'stim.shape = {stim.shape}')
        
        # loop over
        for step in range(self.timesteps):
            x = stim[step, 0, :]*torch.ones(self.inp_1)
            y = stim[step, 1, :]*torch.ones(self.inp_1)
            
            stim_1 = torch.sqrt(x[0]**2 + y[0]**2)*(x > self.inp_1_offsets)*(y > self.inp_1_offsets) # first quadrant
            cur_in_1 = self.fc_in_1(stim_1)
            spk_in_1, pre_in_1, post_in_1, mem_in_1 = self.lif_in_1(cur_in_1,pre_in_1,post_in_1,mem_in_1)
            
            stim_2 = torch.sqrt(x[0]**2 + y[0]**2)*(x < self.inp_2_offsets)*(y > self.inp_2_offsets) # second quadrant
            cur_in_2 = self.fc_in_2(stim_2)
            spk_in_2, pre_in_2, post_in_2, mem_in_2 = self.lif_in_2(cur_in_2,pre_in_2,post_in_2,mem_in_2)
            
            stim_3 = torch.sqrt(x[0]**2 + y[0]**2)*(x < self.inp_3_offsets)*(y < self.inp_3_offsets) # third quadrant
            cur_in_3 = self.fc_in_3(stim_3)
            spk_in_3, pre_in_3, post_in_3, mem_in_3 = self.lif_in_3(cur_in_3,pre_in_3,post_in_3,mem_in_3)
            
            stim_4 = torch.sqrt(x[0]**2 + y[0]**2)*(x > self.inp_4_offsets)*(y < self.inp_4_offsets) # fourth quadrant
            cur_in_4 = self.fc_in_4(stim_4)
            spk_in_4, pre_in_4, post_in_4, mem_in_4 = self.lif_in_4(cur_in_4,pre_in_4,post_in_4,mem_in_4)
            
            cur_h_1 = self.fc_h_1(spk_in_1)
            cur_h_2 = self.fc_h_2(spk_in_2)
            cur_h_3 = self.fc_h_3(spk_in_3)
            cur_h_4 = self.fc_h_4(spk_in_4)
            
            cur_h = cur_h_1 + cur_h_2 + cur_h_3 + cur_h_4
            spk_h, presyn_h, postsyn_h, mem_h = self.lif_h(cur_h, presyn_h, postsyn_h, mem_h)
            
            cur_out = self.fc_out(spk_h)
            spk_out, presyn_out, postsyn_out, mem_out = self.lif_out(cur_out, presyn_out, postsyn_out, mem_out)
            
#             torchaudio.functional.convolve(spk_nrn0,kernel)
#             countVec = torch.tensor([torchaudio.functional.convolve(spk_out[:,i],kernel,mode='valid') for i in range(spk_out_rec.shape[1])])
            
#             print(f'spk_out.shape = {spk_out.shape}')
#             readout = torch.matmul(self.readout_vec,spk_out)
#             print(f'readout.shape = {readout.shape}')
#             cur_out = self.fc_out(spk_out)
            
            cur_readout = self.fc_readout(spk_out)
            _, mem_readout = self.li_readout(cur_readout, mem_readout)
            
#             spk_in_pos_rec.append(spk_in_pos)
#             spk_in_neg_rec.append(spk_in_neg)
            spk_in_1_rec.append(spk_in_1)
            spk_in_2_rec.append(spk_in_2)
            spk_in_3_rec.append(spk_in_3)
            spk_in_4_rec.append(spk_in_4)
            spk_h_rec.append(spk_h)
            spk_out_rec.append(spk_out)
            mem_readout_rec.append(mem_readout)
#             readout_rec.append(readout)
            
            
        mem_readout_rec = torch.stack(mem_readout_rec)
        mem_new = mem_readout_rec.view(self.timesteps, 2, 1)
#         readout_rec = torch.stack(readout_rec)
#         readout_new = readout_rec.view(self.timesteps, 2, 1)
#         spk_in_pos_rec = torch.stack(spk_in_pos_rec)
#         spk_in_neg_rec = torch.stack(spk_in_neg_rec)
        spk_in_1_rec = torch.stack(spk_in_1_rec)
        spk_in_2_rec = torch.stack(spk_in_2_rec)
        spk_in_3_rec = torch.stack(spk_in_3_rec)
        spk_in_4_rec = torch.stack(spk_in_4_rec)
        spk_h_rec = torch.stack(spk_h_rec)
        spk_out_rec = torch.stack(spk_out_rec)
#         countVec = torch.tensor([torchaudio.functional.convolve(spk_out_rec[:,i],kernel,mode='valid') for i in range(spk_out_rec.shape[1])])
        countVec = torchaudio.functional.convolve(torch.t(spk_out_rec),kernel,mode='valid')
        num_times = countVec.shape[1]
        readout_rec = torch.matmul(torch.t(countVec),self.readout_vec)
#         print(f'readout_rec.shape = {readout_rec.shape}')
        readout = readout_rec.view(num_times, 2, 1)
    
#         countVec = 
#         print(f'countVec.shape = {countVec.shape}')
        
        
        return [readout, mem_new, spk_in_1_rec, spk_in_2_rec, spk_in_3_rec, spk_in_4_rec, spk_h_rec, spk_out_rec]
    
    
class ConDivNet2dStimSparse_countTimeReadout(torch.nn.Module):
    """spiking neural network in snntorch."""

    def __init__(self, timesteps, Nin, Nh, Nout, pcon, alpha):
        super().__init__()
        
        self.timesteps = timesteps # number of time steps to simulate the network
        self.inp_1 = int(Nin/4)
        self.inp_2 = int(Nin/4)
        self.inp_3 = int(Nin/4)
        self.inp_4 = int(Nin/4)
        self.h = Nh # number of hidden neurons 
        self.out = Nout
        self.p = pcon # connection sparsity (% of connections to set to zero)
        self.alpha = alpha
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function
        self.inp_1_offsets = 0.1*torch.randn(self.inp_1)
        self.inp_2_offsets = 0.1*torch.randn(self.inp_2)
        self.inp_3_offsets = 0.1*torch.randn(self.inp_3)
        self.inp_4_offsets = 0.1*torch.randn(self.inp_4)
        
        self.time_readout_vec = torch.nn.Parameter(0.1*torch.randn(Nout,2))
        self.count_readout_vec = torch.nn.Parameter(0.1*torch.randn(Nout,2))
#         print(f'self.readout_vec.shape = {self.readout_vec.shape}')
        
        # global parameters
        alpha_min = 0.7
        alpha_max = 0.9
        thr_min = 0.0
        thr_max = 0.5
        
##################################################################################################
        
        # stimulus to quadrant 1 input layer
        self.fc_in_1 = torch.nn.Linear(in_features=self.inp_1, out_features=self.inp_1, bias=False)
#         torch.nn.init.sparse_(self.fc_in_1.weight, sparsity=self.p, std=1./np.sqrt(1*self.p))
        # randomly initialize parameters for input layer
        alpha_in_1 = (alpha_max-alpha_min)*torch.rand(self.inp_1) + alpha_min
        beta_in_1 = alpha_in_1 - 0.1
        thr_in_1 = torch.rand(self.inp_1)*(thr_max-thr_min) + thr_min
        self.lif_in_1 = snn.Alpha(alpha=alpha_in_1, beta=beta_in_1, 
                                threshold=thr_in_1, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 2 input layer
        self.fc_in_2 = torch.nn.Linear(in_features=self.inp_2, out_features=self.inp_2, bias=False)
#         torch.nn.init.sparse_(self.fc_in_2.weight, sparsity=self.p, std=1./np.sqrt(1*self.p))
        # randomly initialize parameters for input layer
        alpha_in_2 = (alpha_max-alpha_min)*torch.rand(self.inp_2) + alpha_min
        beta_in_2 = alpha_in_2 - 0.1
        thr_in_2 = torch.rand(self.inp_2)*(thr_max-thr_min) + thr_min
        self.lif_in_2 = snn.Alpha(alpha=alpha_in_2, beta=beta_in_2, 
                                threshold=thr_in_2, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 3 input layer
        self.fc_in_3 = torch.nn.Linear(in_features=self.inp_3, out_features=self.inp_3, bias=False)
#         torch.nn.init.sparse_(self.fc_in_3.weight, sparsity=self.p, std=1.)
        # randomly initialize parameters for input layer
        alpha_in_3 = (alpha_max-alpha_min)*torch.rand(self.inp_3) + alpha_min
        beta_in_3 = alpha_in_3 - 0.1
        thr_in_3 = torch.rand(self.inp_3)*(thr_max-thr_min) + thr_min
        self.lif_in_3 = snn.Alpha(alpha=alpha_in_3, beta=beta_in_3, 
                                threshold=thr_in_3, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 4 input layer
        self.fc_in_4 = torch.nn.Linear(in_features=self.inp_4, out_features=self.inp_4, bias=False)
#         torch.nn.init.sparse_(self.fc_in_4.weight, sparsity=self.p, std=1.)
        # randomly initialize parameters for input layer
        alpha_in_4 = (alpha_max-alpha_min)*torch.rand(self.inp_4) + alpha_min
        beta_in_4 = alpha_in_4 - 0.1
        thr_in_4 = torch.rand(self.inp_4)*(thr_max-thr_min) + thr_min
        self.lif_in_4 = snn.Alpha(alpha=alpha_in_4, beta=beta_in_4, 
                                threshold=thr_in_4, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        
        # quadrant 1 input layer to E1 layer
        self.fc_h_1 = torch.nn.Linear(in_features=self.inp_1, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_1.weight, sparsity=self.p, std=1./np.sqrt(self.inp_1*(1-self.p)))
        # quadrant 2 input layer to E1 layer
        self.fc_h_2 = torch.nn.Linear(in_features=self.inp_2, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_2.weight, sparsity=self.p, std=1./np.sqrt(self.inp_2*(1-self.p)))
        # quadrant 1 input layer to E1 layer
        self.fc_h_3 = torch.nn.Linear(in_features=self.inp_3, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_3.weight, sparsity=self.p, std=1./np.sqrt(self.inp_3*(1-self.p)))
        # quadrant 2 input layer to E1 layer
        self.fc_h_4 = torch.nn.Linear(in_features=self.inp_4, out_features=self.h, bias=False) 
        torch.nn.init.sparse_(self.fc_h_4.weight, sparsity=self.p, std=1./np.sqrt(self.inp_4*(1-self.p)))
        # randomly initialize parameters for E1 layer
        alpha_h = (alpha_max-alpha_min)*torch.rand(self.h) + alpha_min
        beta_h = alpha_h - 0.1
        thr_h = torch.rand(self.h)*(thr_max-thr_min) + thr_min
        self.lif_h = snn.Alpha(alpha=alpha_h, beta=beta_h, threshold=thr_h, 
                                learn_threshold=False, spike_grad=spike_grad)
        
##################################################################################################

        # hidden layer to output layer
        self.fc_out = torch.nn.Linear(in_features=self.h, out_features=self.out,bias=False)
        torch.nn.init.sparse_(self.fc_out.weight, sparsity=self.p, std=1./np.sqrt(self.h*(1-self.p)))
        # randomly initialize parameters for B layer
        alpha_out = (alpha_max-alpha_min)*torch.rand(self.out) + alpha_min
        beta_out = alpha_out - 0.1
        thr_out = torch.rand(self.out)*(thr_max-thr_min) + thr_min
        self.lif_out = snn.Alpha(alpha=alpha_out, beta=beta_out, threshold=thr_out, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # output layer to readout neuron
        self.fc_readout = torch.nn.Linear(in_features=self.out, out_features=2,bias=False)        
        # randomly initialize decay rate for single output neuron
        thr_readout = torch.ones(2)
        beta_readout = torch.rand(2)
        # leaky integrator readout neuron used for reconstruction
        self.li_readout = snn.Leaky(beta=beta_readout, threshold=thr_readout, learn_beta=True, 
                                    spike_grad=spike_grad, reset_mechanism="none")
        

    def forward(self, stim):
        """Forward pass for several time steps."""

        # Initalize membrane potential
#         pre_in_pos, post_in_pos, mem_in_pos = self.lif_in_pos.init_alpha()
#         pre_in_neg, post_in_neg, mem_in_neg = self.lif_in_neg.init_alpha()
        pre_in_1, post_in_1, mem_in_1 = self.lif_in_1.init_alpha()
        pre_in_2, post_in_2, mem_in_2 = self.lif_in_2.init_alpha()
        pre_in_3, post_in_3, mem_in_3 = self.lif_in_3.init_alpha()
        pre_in_4, post_in_4, mem_in_4 = self.lif_in_4.init_alpha()
        presyn_h, postsyn_h, mem_h = self.lif_h.init_alpha()
        presyn_out, postsyn_out, mem_out = self.lif_out.init_alpha()
        mem_readout = self.li_readout.init_leaky()

        # Empty lists to record outputs
        spk_in_1_rec = []
        spk_in_2_rec = []
        spk_in_3_rec = []
        spk_in_4_rec = []
        spk_h_rec = []
        spk_out_rec = []
        mem_readout_rec = []
#         readout_rec = []
        
        deltatt = 10
        kernelRange = torch.linspace(0,1,deltatt+1)
        sigma = 0.1
        half = kernelRange[int(deltatt/2)]
        k = torch.exp(-((kernelRange-half)**2)/sigma)
        time_kernel = torch.zeros((self.out,deltatt+1))
        for i in range(self.out):
            time_kernel[i,:] = k
            
        deltatc = 70
        kernelRange = torch.linspace(0,1,deltatc+1)
        sigma = 0.1
        half = kernelRange[int(deltatc/2)]
        k = torch.exp(-((kernelRange-half)**2)/sigma)
        count_kernel = torch.zeros((self.out,deltatc+1))
        for i in range(self.out):
            count_kernel[i,:] = k
#         kernel = torch.tensor([torch.exp(-((x-half)**2)/sigma) for x in kernelRange])
#         plt.plot(kernel)
#         plt.show()
#         plt.close()
        
        # print(f'stim.shape = {stim.shape}')
        
        # loop over
        for step in range(self.timesteps):
            x = stim[step, 0, :]*torch.ones(self.inp_1)
            y = stim[step, 1, :]*torch.ones(self.inp_1)
            
            stim_1 = torch.sqrt(x[0]**2 + y[0]**2)*(x > self.inp_1_offsets)*(y > self.inp_1_offsets) # first quadrant
            cur_in_1 = self.fc_in_1(stim_1)
            spk_in_1, pre_in_1, post_in_1, mem_in_1 = self.lif_in_1(cur_in_1,pre_in_1,post_in_1,mem_in_1)
            
            stim_2 = torch.sqrt(x[0]**2 + y[0]**2)*(x < self.inp_2_offsets)*(y > self.inp_2_offsets) # second quadrant
            cur_in_2 = self.fc_in_2(stim_2)
            spk_in_2, pre_in_2, post_in_2, mem_in_2 = self.lif_in_2(cur_in_2,pre_in_2,post_in_2,mem_in_2)
            
            stim_3 = torch.sqrt(x[0]**2 + y[0]**2)*(x < self.inp_3_offsets)*(y < self.inp_3_offsets) # third quadrant
            cur_in_3 = self.fc_in_3(stim_3)
            spk_in_3, pre_in_3, post_in_3, mem_in_3 = self.lif_in_3(cur_in_3,pre_in_3,post_in_3,mem_in_3)
            
            stim_4 = torch.sqrt(x[0]**2 + y[0]**2)*(x > self.inp_4_offsets)*(y < self.inp_4_offsets) # fourth quadrant
            cur_in_4 = self.fc_in_4(stim_4)
            spk_in_4, pre_in_4, post_in_4, mem_in_4 = self.lif_in_4(cur_in_4,pre_in_4,post_in_4,mem_in_4)
            
            cur_h_1 = self.fc_h_1(spk_in_1)
            cur_h_2 = self.fc_h_2(spk_in_2)
            cur_h_3 = self.fc_h_3(spk_in_3)
            cur_h_4 = self.fc_h_4(spk_in_4)
            
            cur_h = cur_h_1 + cur_h_2 + cur_h_3 + cur_h_4
            spk_h, presyn_h, postsyn_h, mem_h = self.lif_h(cur_h, presyn_h, postsyn_h, mem_h)
            
            cur_out = self.fc_out(spk_h)
            spk_out, presyn_out, postsyn_out, mem_out = self.lif_out(cur_out, presyn_out, postsyn_out, mem_out)
            
#             torchaudio.functional.convolve(spk_nrn0,kernel)
#             countVec = torch.tensor([torchaudio.functional.convolve(spk_out[:,i],kernel,mode='valid') for i in range(spk_out_rec.shape[1])])
            
#             print(f'spk_out.shape = {spk_out.shape}')
#             readout = torch.matmul(self.readout_vec,spk_out)
#             print(f'readout.shape = {readout.shape}')
#             cur_out = self.fc_out(spk_out)
            
            cur_readout = self.fc_readout(spk_out)
            _, mem_readout = self.li_readout(cur_readout, mem_readout)
            
#             spk_in_pos_rec.append(spk_in_pos)
#             spk_in_neg_rec.append(spk_in_neg)
            spk_in_1_rec.append(spk_in_1)
            spk_in_2_rec.append(spk_in_2)
            spk_in_3_rec.append(spk_in_3)
            spk_in_4_rec.append(spk_in_4)
            spk_h_rec.append(spk_h)
            spk_out_rec.append(spk_out)
            mem_readout_rec.append(mem_readout)
#             readout_rec.append(readout)
            
            
        mem_readout_rec = torch.stack(mem_readout_rec)
        mem_new = mem_readout_rec.view(self.timesteps, 2, 1)
#         readout_rec = torch.stack(readout_rec)
#         readout_new = readout_rec.view(self.timesteps, 2, 1)
#         spk_in_pos_rec = torch.stack(spk_in_pos_rec)
#         spk_in_neg_rec = torch.stack(spk_in_neg_rec)
        spk_in_1_rec = torch.stack(spk_in_1_rec)
        spk_in_2_rec = torch.stack(spk_in_2_rec)
        spk_in_3_rec = torch.stack(spk_in_3_rec)
        spk_in_4_rec = torch.stack(spk_in_4_rec)
        spk_h_rec = torch.stack(spk_h_rec)
        spk_out_rec = torch.stack(spk_out_rec)

        timevec = torchaudio.functional.convolve(torch.t(spk_out_rec),time_kernel,mode='valid')
        countvec = torchaudio.functional.convolve(torch.t(spk_out_rec),count_kernel,mode='valid')
        yhat_time_pre = torch.matmul(torch.t(timevec),self.time_readout_vec)
        yhat_count = torch.matmul(torch.t(countvec),self.count_readout_vec)
        tmax = yhat_count.shape[0]
        yhat_time = yhat_time_pre[:tmax,:]
#         print(f'yhat_count.shape = {yhat_count.shape}')
#         print(f'yhat_time.shape = {yhat_time.shape}')
        readout_rec = self.alpha*yhat_time + (1-self.alpha)*yhat_count
        
#         yhat_time = torch.reshape(yhat_time,yhat_count.shape)
#         yhat = self.alpha*yhat_time
        
#         num_times = countVec.shape[1]
#         readout_rec = torch.matmul(torch.t(countVec),self.readout_vec)
#         print(f'readout_rec.shape = {readout_rec.shape}')
        readout = readout_rec.view(tmax, 2, 1)
#         readout = 0
#         countVec = 
#         print(f'countVec.shape = {countVec.shape}')
        
        
        return [readout, mem_new, spk_in_1_rec, spk_in_2_rec, spk_in_3_rec, spk_in_4_rec, spk_h_rec, spk_out_rec]
    
class LIFConDivNet2dStimSparse_countTimeReadout(torch.nn.Module):
    """spiking neural network in snntorch."""

    def __init__(self, timesteps, Nin, Nh, Nout, pcon, alpha):
        super().__init__()
        
        self.timesteps = timesteps # number of time steps to simulate the network
        self.inp_1 = int(Nin/4)
        self.inp_2 = int(Nin/4)
        self.inp_3 = int(Nin/4)
        self.inp_4 = int(Nin/4)
        self.h = Nh # number of hidden neurons 
        self.out = Nout
        self.p = pcon # connection sparsity (% of connections to set to zero)
        self.alpha = alpha
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function
        self.inp_1_offsets = 0.1*torch.randn(self.inp_1)
        self.inp_2_offsets = 0.1*torch.randn(self.inp_2)
        self.inp_3_offsets = 0.1*torch.randn(self.inp_3)
        self.inp_4_offsets = 0.1*torch.randn(self.inp_4)
        
        self.time_readout_vec = torch.nn.Parameter(0.1*torch.randn(Nout,2))
        self.count_readout_vec = torch.nn.Parameter(0.1*torch.randn(Nout,2))
#         print(f'self.readout_vec.shape = {self.readout_vec.shape}')
        
        # global parameters
        beta_min = 0.7
        beta_max = 0.9
        thr_min = 0.0
        thr_max = 0.5
        
##################################################################################################
        
        # stimulus to quadrant 1 input layer
        self.fc_in_1 = torch.nn.Linear(in_features=self.inp_1, out_features=self.inp_1, bias=False)
        beta_in_1 = (beta_max-beta_min)*torch.rand(self.inp_1) + beta_min
        thr_in_1 = torch.rand(self.inp_1)*(thr_max-thr_min) + thr_min
        self.lif_in_1 = snn.Leaky(beta=beta_in_1, 
                                threshold=thr_in_1, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 2 input layer
        self.fc_in_2 = torch.nn.Linear(in_features=self.inp_2, out_features=self.inp_2, bias=False)
        beta_in_2 = (beta_max-beta_min)*torch.rand(self.inp_1) + beta_min
        thr_in_2 = torch.rand(self.inp_2)*(thr_max-thr_min) + thr_min
        self.lif_in_2 = snn.Leaky(beta=beta_in_2, 
                                threshold=thr_in_2, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 3 input layer
        self.fc_in_3 = torch.nn.Linear(in_features=self.inp_3, out_features=self.inp_3, bias=False)
        beta_in_3 = (beta_max-beta_min)*torch.rand(self.inp_1) + beta_min
        thr_in_3 = torch.rand(self.inp_3)*(thr_max-thr_min) + thr_min
        self.lif_in_3 = snn.Leaky(beta=beta_in_3, 
                                threshold=thr_in_3, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 4 input layer
        self.fc_in_4 = torch.nn.Linear(in_features=self.inp_4, out_features=self.inp_4, bias=False)
        beta_in_4 = (beta_max-beta_min)*torch.rand(self.inp_1) + beta_min
        thr_in_4 = torch.rand(self.inp_4)*(thr_max-thr_min) + thr_min
        self.lif_in_4 = snn.Leaky(beta=beta_in_4, 
                                threshold=thr_in_4, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        
        # quadrant 1 input layer to E1 layer
        self.fc_h_1 = torch.nn.Linear(in_features=self.inp_1, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_1.weight, sparsity=self.p, std=1./np.sqrt(self.inp_1*(1-self.p)))
        # quadrant 2 input layer to E1 layer
        self.fc_h_2 = torch.nn.Linear(in_features=self.inp_2, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_2.weight, sparsity=self.p, std=1./np.sqrt(self.inp_2*(1-self.p)))
        # quadrant 1 input layer to E1 layer
        self.fc_h_3 = torch.nn.Linear(in_features=self.inp_3, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_3.weight, sparsity=self.p, std=1./np.sqrt(self.inp_3*(1-self.p)))
        # quadrant 2 input layer to E1 layer
        self.fc_h_4 = torch.nn.Linear(in_features=self.inp_4, out_features=self.h, bias=False) 
        torch.nn.init.sparse_(self.fc_h_4.weight, sparsity=self.p, std=1./np.sqrt(self.inp_4*(1-self.p)))
        # randomly initialize parameters for E1 layer
        beta_h = (beta_max-beta_min)*torch.rand(self.h) + beta_min
        thr_h = torch.rand(self.h)*(thr_max-thr_min) + thr_min
        self.lif_h = snn.Leaky(beta=beta_h, threshold=thr_h, 
                                learn_threshold=False, spike_grad=spike_grad)
        
##################################################################################################

        # hidden layer to output layer
        self.fc_out = torch.nn.Linear(in_features=self.h, out_features=self.out,bias=False)
        torch.nn.init.sparse_(self.fc_out.weight, sparsity=self.p, std=1./np.sqrt(self.h*(1-self.p)))
        # randomly initialize parameters for B layer
        beta_out = (beta_max-beta_min)*torch.rand(self.out) + beta_min
        thr_out = torch.rand(self.out)*(thr_max-thr_min) + thr_min
        self.lif_out = snn.Leaky(beta=beta_out, threshold=thr_out, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # output layer to readout neuron
        self.fc_readout = torch.nn.Linear(in_features=self.out, out_features=2,bias=False)        
        # randomly initialize decay rate for single output neuron
        thr_readout = torch.ones(2)
        beta_readout = torch.rand(2)
        # leaky integrator readout neuron used for reconstruction
        self.li_readout = snn.Leaky(beta=beta_readout, threshold=thr_readout, learn_beta=True, 
                                    spike_grad=spike_grad, reset_mechanism="none")
        

    def forward(self, stim):
        """Forward pass for several time steps."""

        # Initalize membrane potential
#         pre_in_pos, post_in_pos, mem_in_pos = self.lif_in_pos.init_alpha()
#         pre_in_neg, post_in_neg, mem_in_neg = self.lif_in_neg.init_alpha()
        pre_in_1, post_in_1, mem_in_1 = self.lif_in_1.init_alpha()
        pre_in_2, post_in_2, mem_in_2 = self.lif_in_2.init_alpha()
        pre_in_3, post_in_3, mem_in_3 = self.lif_in_3.init_alpha()
        pre_in_4, post_in_4, mem_in_4 = self.lif_in_4.init_alpha()
        presyn_h, postsyn_h, mem_h = self.lif_h.init_alpha()
        presyn_out, postsyn_out, mem_out = self.lif_out.init_alpha()
        mem_readout = self.li_readout.init_leaky()

        # Empty lists to record outputs
        spk_in_1_rec = []
        spk_in_2_rec = []
        spk_in_3_rec = []
        spk_in_4_rec = []
        spk_h_rec = []
        spk_out_rec = []
        mem_readout_rec = []
#         readout_rec = []
        
        deltatt = 10
        kernelRange = torch.linspace(0,1,deltatt+1)
        sigma = 0.1
        half = kernelRange[int(deltatt/2)]
        k = torch.exp(-((kernelRange-half)**2)/sigma)
        time_kernel = torch.zeros((self.out,deltatt+1))
        for i in range(self.out):
            time_kernel[i,:] = k
            
        deltatc = 70
        kernelRange = torch.linspace(0,1,deltatc+1)
        sigma = 0.1
        half = kernelRange[int(deltatc/2)]
        k = torch.exp(-((kernelRange-half)**2)/sigma)
        count_kernel = torch.zeros((self.out,deltatc+1))
        for i in range(self.out):
            count_kernel[i,:] = k
#         kernel = torch.tensor([torch.exp(-((x-half)**2)/sigma) for x in kernelRange])
#         plt.plot(kernel)
#         plt.show()
#         plt.close()
        
        # print(f'stim.shape = {stim.shape}')
        
        # loop over
        for step in range(self.timesteps):
            x = stim[step, 0, :]*torch.ones(self.inp_1)
            y = stim[step, 1, :]*torch.ones(self.inp_1)
            
            stim_1 = torch.sqrt(x[0]**2 + y[0]**2)*(x > self.inp_1_offsets)*(y > self.inp_1_offsets) # first quadrant
            cur_in_1 = self.fc_in_1(stim_1)
            spk_in_1, mem_in_1 = self.lif_in_1(cur_in_1,mem_in_1)
            
            stim_2 = torch.sqrt(x[0]**2 + y[0]**2)*(x < self.inp_2_offsets)*(y > self.inp_2_offsets) # second quadrant
            cur_in_2 = self.fc_in_2(stim_2)
            spk_in_2, mem_in_2 = self.lif_in_2(cur_in_2,mem_in_2)
            
            stim_3 = torch.sqrt(x[0]**2 + y[0]**2)*(x < self.inp_3_offsets)*(y < self.inp_3_offsets) # third quadrant
            cur_in_3 = self.fc_in_3(stim_3)
            spk_in_3, mem_in_3 = self.lif_in_3(cur_in_3,mem_in_3)
            
            stim_4 = torch.sqrt(x[0]**2 + y[0]**2)*(x > self.inp_4_offsets)*(y < self.inp_4_offsets) # fourth quadrant
            cur_in_4 = self.fc_in_4(stim_4)
            spk_in_4, mem_in_4 = self.lif_in_4(cur_in_4,mem_in_4)
            
            cur_h_1 = self.fc_h_1(spk_in_1)
            cur_h_2 = self.fc_h_2(spk_in_2)
            cur_h_3 = self.fc_h_3(spk_in_3)
            cur_h_4 = self.fc_h_4(spk_in_4)
            
            cur_h = cur_h_1 + cur_h_2 + cur_h_3 + cur_h_4
            spk_h, mem_h = self.lif_h(cur_h, mem_h)
            
            cur_out = self.fc_out(spk_h)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            
#             torchaudio.functional.convolve(spk_nrn0,kernel)
#             countVec = torch.tensor([torchaudio.functional.convolve(spk_out[:,i],kernel,mode='valid') for i in range(spk_out_rec.shape[1])])
            
#             print(f'spk_out.shape = {spk_out.shape}')
#             readout = torch.matmul(self.readout_vec,spk_out)
#             print(f'readout.shape = {readout.shape}')
#             cur_out = self.fc_out(spk_out)
            
            cur_readout = self.fc_readout(spk_out)
            _, mem_readout = self.li_readout(cur_readout, mem_readout)
            
#             spk_in_pos_rec.append(spk_in_pos)
#             spk_in_neg_rec.append(spk_in_neg)
            spk_in_1_rec.append(spk_in_1)
            spk_in_2_rec.append(spk_in_2)
            spk_in_3_rec.append(spk_in_3)
            spk_in_4_rec.append(spk_in_4)
            spk_h_rec.append(spk_h)
            spk_out_rec.append(spk_out)
            mem_readout_rec.append(mem_readout)
#             readout_rec.append(readout)
            
            
        mem_readout_rec = torch.stack(mem_readout_rec)
        mem_new = mem_readout_rec.view(self.timesteps, 2, 1)
#         readout_rec = torch.stack(readout_rec)
#         readout_new = readout_rec.view(self.timesteps, 2, 1)
#         spk_in_pos_rec = torch.stack(spk_in_pos_rec)
#         spk_in_neg_rec = torch.stack(spk_in_neg_rec)
        spk_in_1_rec = torch.stack(spk_in_1_rec)
        spk_in_2_rec = torch.stack(spk_in_2_rec)
        spk_in_3_rec = torch.stack(spk_in_3_rec)
        spk_in_4_rec = torch.stack(spk_in_4_rec)
        spk_h_rec = torch.stack(spk_h_rec)
        spk_out_rec = torch.stack(spk_out_rec)

        timevec = torchaudio.functional.convolve(torch.t(spk_out_rec),time_kernel,mode='valid')
        countvec = torchaudio.functional.convolve(torch.t(spk_out_rec),count_kernel,mode='valid')
        yhat_time_pre = torch.matmul(torch.t(timevec),self.time_readout_vec)
        yhat_count = torch.matmul(torch.t(countvec),self.count_readout_vec)
        tmax = yhat_count.shape[0]
        yhat_time = yhat_time_pre[:tmax,:]
#         print(f'yhat_count.shape = {yhat_count.shape}')
#         print(f'yhat_time.shape = {yhat_time.shape}')
        readout_rec = self.alpha*yhat_time + (1-self.alpha)*yhat_count
        
#         yhat_time = torch.reshape(yhat_time,yhat_count.shape)
#         yhat = self.alpha*yhat_time
        
#         num_times = countVec.shape[1]
#         readout_rec = torch.matmul(torch.t(countVec),self.readout_vec)
#         print(f'readout_rec.shape = {readout_rec.shape}')
        readout = readout_rec.view(tmax, 2, 1)
#         readout = 0
#         countVec = 
#         print(f'countVec.shape = {countVec.shape}')
        
        
        return [readout, mem_new, spk_in_1_rec, spk_in_2_rec, spk_in_3_rec, spk_in_4_rec, spk_h_rec, spk_out_rec]
    
    
class ConDivNet2dStimSparse(torch.nn.Module):
    """spiking neural network in snntorch."""

    def __init__(self, timesteps, Nin, Nh, Nout, pcon):
        super().__init__()
        
        self.timesteps = timesteps # number of time steps to simulate the network
        self.inp_1 = int(Nin/4)
        self.inp_2 = int(Nin/4)
        self.inp_3 = int(Nin/4)
        self.inp_4 = int(Nin/4)
        self.h = Nh # number of hidden neurons 
        self.out = Nout
        self.p = pcon # connection sparsity (% of connections to set to zero)
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function
        self.inp_1_offsets = 0.1*torch.randn(self.inp_1)
        self.inp_2_offsets = 0.1*torch.randn(self.inp_2)
        self.inp_3_offsets = 0.1*torch.randn(self.inp_3)
        self.inp_4_offsets = 0.1*torch.randn(self.inp_4)
        
        # global parameters
        alpha_min = 0.7
        alpha_max = 0.9
        thr_min = 0.0
        thr_max = 0.5
        
##################################################################################################
        
        # stimulus to quadrant 1 input layer
        self.fc_in_1 = torch.nn.Linear(in_features=self.inp_1, out_features=self.inp_1, bias=False)
#         torch.nn.init.sparse_(self.fc_in_1.weight, sparsity=self.p, std=1./np.sqrt(1*self.p))
        # randomly initialize parameters for input layer
        alpha_in_1 = (alpha_max-alpha_min)*torch.rand(self.inp_1) + alpha_min
        beta_in_1 = alpha_in_1 - 0.1
        thr_in_1 = torch.rand(self.inp_1)*(thr_max-thr_min) + thr_min
        self.lif_in_1 = snn.Alpha(alpha=alpha_in_1, beta=beta_in_1, 
                                threshold=thr_in_1, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 2 input layer
        self.fc_in_2 = torch.nn.Linear(in_features=self.inp_2, out_features=self.inp_2, bias=False)
#         torch.nn.init.sparse_(self.fc_in_2.weight, sparsity=self.p, std=1./np.sqrt(1*self.p))
        # randomly initialize parameters for input layer
        alpha_in_2 = (alpha_max-alpha_min)*torch.rand(self.inp_2) + alpha_min
        beta_in_2 = alpha_in_2 - 0.1
        thr_in_2 = torch.rand(self.inp_2)*(thr_max-thr_min) + thr_min
        self.lif_in_2 = snn.Alpha(alpha=alpha_in_2, beta=beta_in_2, 
                                threshold=thr_in_2, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 3 input layer
        self.fc_in_3 = torch.nn.Linear(in_features=self.inp_3, out_features=self.inp_3, bias=False)
#         torch.nn.init.sparse_(self.fc_in_3.weight, sparsity=self.p, std=1.)
        # randomly initialize parameters for input layer
        alpha_in_3 = (alpha_max-alpha_min)*torch.rand(self.inp_3) + alpha_min
        beta_in_3 = alpha_in_3 - 0.1
        thr_in_3 = torch.rand(self.inp_3)*(thr_max-thr_min) + thr_min
        self.lif_in_3 = snn.Alpha(alpha=alpha_in_3, beta=beta_in_3, 
                                threshold=thr_in_3, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 4 input layer
        self.fc_in_4 = torch.nn.Linear(in_features=self.inp_4, out_features=self.inp_4, bias=False)
#         torch.nn.init.sparse_(self.fc_in_4.weight, sparsity=self.p, std=1.)
        # randomly initialize parameters for input layer
        alpha_in_4 = (alpha_max-alpha_min)*torch.rand(self.inp_4) + alpha_min
        beta_in_4 = alpha_in_4 - 0.1
        thr_in_4 = torch.rand(self.inp_4)*(thr_max-thr_min) + thr_min
        self.lif_in_4 = snn.Alpha(alpha=alpha_in_4, beta=beta_in_4, 
                                threshold=thr_in_4, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        
        # quadrant 1 input layer to E1 layer
        self.fc_h_1 = torch.nn.Linear(in_features=self.inp_1, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_1.weight, sparsity=self.p, std=1./np.sqrt(self.inp_1*(1-self.p)))
        # quadrant 2 input layer to E1 layer
        self.fc_h_2 = torch.nn.Linear(in_features=self.inp_2, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_2.weight, sparsity=self.p, std=1./np.sqrt(self.inp_2*(1-self.p)))
        # quadrant 1 input layer to E1 layer
        self.fc_h_3 = torch.nn.Linear(in_features=self.inp_3, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_3.weight, sparsity=self.p, std=1./np.sqrt(self.inp_3*(1-self.p)))
        # quadrant 2 input layer to E1 layer
        self.fc_h_4 = torch.nn.Linear(in_features=self.inp_4, out_features=self.h, bias=False) 
        torch.nn.init.sparse_(self.fc_h_4.weight, sparsity=self.p, std=1./np.sqrt(self.inp_4*(1-self.p)))
        # randomly initialize parameters for E1 layer
        alpha_h = (alpha_max-alpha_min)*torch.rand(self.h) + alpha_min
        beta_h = alpha_h - 0.1
        thr_h = torch.rand(self.h)*(thr_max-thr_min) + thr_min
        self.lif_h = snn.Alpha(alpha=alpha_h, beta=beta_h, threshold=thr_h, 
                                learn_threshold=False, spike_grad=spike_grad)
        
##################################################################################################

        # hidden layer to output layer
        self.fc_out = torch.nn.Linear(in_features=self.h, out_features=self.out,bias=False)
        torch.nn.init.sparse_(self.fc_out.weight, sparsity=self.p, std=1./np.sqrt(self.h*(1-self.p)))
        # randomly initialize parameters for B layer
        alpha_out = (alpha_max-alpha_min)*torch.rand(self.out) + alpha_min
        beta_out = alpha_out - 0.1
        thr_out = torch.rand(self.out)*(thr_max-thr_min) + thr_min
        self.lif_out = snn.Alpha(alpha=alpha_out, beta=beta_out, threshold=thr_out, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # output layer to readout neuron
        self.fc_readout = torch.nn.Linear(in_features=self.out, out_features=2,bias=False)        
        # randomly initialize decay rate for single output neuron
        thr_readout = torch.ones(2)
        beta_readout = torch.rand(2)
        # leaky integrator readout neuron used for reconstruction
        self.li_readout = snn.Leaky(beta=beta_readout, threshold=thr_readout, learn_beta=True, 
                                    spike_grad=spike_grad, reset_mechanism="none")
        
#         # randomly initialize threshold for input layer
#         alpha_out = (alpha_max-alpha_min)*torch.rand(self.out) + alpha_min
#         beta_out = alpha_out - 0.1
#         thr_out = torch.rand(self.out)*(thr_max-thr_min) + thr_min

#         # output layer
#         self.lif_out = snn.Alpha(alpha=alpha_out, beta=beta_out, 
#                                 threshold=thr_out, 
#                                 learn_threshold=False, spike_grad=spike_grad)
#         self.fc_out = torch.nn.Linear(in_features=self.out, out_features=1)
        
#         # randomly initialize decay rate for single output neuron
#         beta_readout = torch.rand(1)
        
#         # leaky integrator readout neuron used for reconstruction
#         self.li_readout = snn.Leaky(beta=beta_readout, threshold=1.0, learn_beta=True, 
#                                     spike_grad=spike_grad, reset_mechanism="none")

    def forward(self, stim):
        """Forward pass for several time steps."""

        # Initalize membrane potential
#         pre_in_pos, post_in_pos, mem_in_pos = self.lif_in_pos.init_alpha()
#         pre_in_neg, post_in_neg, mem_in_neg = self.lif_in_neg.init_alpha()
        pre_in_1, post_in_1, mem_in_1 = self.lif_in_1.init_alpha()
        pre_in_2, post_in_2, mem_in_2 = self.lif_in_2.init_alpha()
        pre_in_3, post_in_3, mem_in_3 = self.lif_in_3.init_alpha()
        pre_in_4, post_in_4, mem_in_4 = self.lif_in_4.init_alpha()
        presyn_h, postsyn_h, mem_h = self.lif_h.init_alpha()
        presyn_out, postsyn_out, mem_out = self.lif_out.init_alpha()
        mem_readout = self.li_readout.init_leaky()

        # Empty lists to record outputs
        spk_in_1_rec = []
        spk_in_2_rec = []
        spk_in_3_rec = []
        spk_in_4_rec = []
        spk_h_rec = []
        spk_out_rec = []
        mem_readout_rec = []
        
        # print(f'stim.shape = {stim.shape}')
        
        # loop over
        for step in range(self.timesteps):
            x = stim[step, 0, :]*torch.ones(self.inp_1)
            y = stim[step, 1, :]*torch.ones(self.inp_1)
            
            stim_1 = torch.sqrt(x[0]**2 + y[0]**2)*(x > self.inp_1_offsets)*(y > self.inp_1_offsets) # first quadrant
            cur_in_1 = self.fc_in_1(stim_1)
            spk_in_1, pre_in_1, post_in_1, mem_in_1 = self.lif_in_1(cur_in_1,pre_in_1,post_in_1,mem_in_1)
            
            stim_2 = torch.sqrt(x[0]**2 + y[0]**2)*(x < self.inp_2_offsets)*(y > self.inp_2_offsets) # second quadrant
            cur_in_2 = self.fc_in_2(stim_2)
            spk_in_2, pre_in_2, post_in_2, mem_in_2 = self.lif_in_2(cur_in_2,pre_in_2,post_in_2,mem_in_2)
            
            stim_3 = torch.sqrt(x[0]**2 + y[0]**2)*(x < self.inp_3_offsets)*(y < self.inp_3_offsets) # third quadrant
            cur_in_3 = self.fc_in_3(stim_3)
            spk_in_3, pre_in_3, post_in_3, mem_in_3 = self.lif_in_3(cur_in_3,pre_in_3,post_in_3,mem_in_3)
            
            stim_4 = torch.sqrt(x[0]**2 + y[0]**2)*(x > self.inp_4_offsets)*(y < self.inp_4_offsets) # fourth quadrant
            cur_in_4 = self.fc_in_4(stim_4)
            spk_in_4, pre_in_4, post_in_4, mem_in_4 = self.lif_in_4(cur_in_4,pre_in_4,post_in_4,mem_in_4)
            
            cur_h_1 = self.fc_h_1(spk_in_1)
            cur_h_2 = self.fc_h_2(spk_in_2)
            cur_h_3 = self.fc_h_3(spk_in_3)
            cur_h_4 = self.fc_h_4(spk_in_4)
            
            cur_h = cur_h_1 + cur_h_2 + cur_h_3 + cur_h_4
            spk_h, presyn_h, postsyn_h, mem_h = self.lif_h(cur_h, presyn_h, postsyn_h, mem_h)
            
            cur_out = self.fc_out(spk_h)
            spk_out, presyn_out, postsyn_out, mem_out = self.lif_out(cur_out, presyn_out, postsyn_out, mem_out)
#             cur_out = self.fc_out(spk_out)
            
            cur_readout = self.fc_readout(spk_out)
            _, mem_readout = self.li_readout(cur_readout, mem_readout)
            
#             spk_in_pos_rec.append(spk_in_pos)
#             spk_in_neg_rec.append(spk_in_neg)
            spk_in_1_rec.append(spk_in_1)
            spk_in_2_rec.append(spk_in_2)
            spk_in_3_rec.append(spk_in_3)
            spk_in_4_rec.append(spk_in_4)
            spk_h_rec.append(spk_h)
            spk_out_rec.append(spk_out)
            mem_readout_rec.append(mem_readout)
            
        mem_readout_rec = torch.stack(mem_readout_rec)
        mem_new = mem_readout_rec.view(self.timesteps, 2, 1)
#         spk_in_pos_rec = torch.stack(spk_in_pos_rec)
#         spk_in_neg_rec = torch.stack(spk_in_neg_rec)
        spk_in_1_rec = torch.stack(spk_in_1_rec)
        spk_in_2_rec = torch.stack(spk_in_2_rec)
        spk_in_3_rec = torch.stack(spk_in_3_rec)
        spk_in_4_rec = torch.stack(spk_in_4_rec)
        spk_h_rec = torch.stack(spk_h_rec)
        spk_out_rec = torch.stack(spk_out_rec)
            
        return [mem_new, spk_in_1_rec, spk_in_2_rec, spk_in_3_rec, spk_in_4_rec, spk_h_rec, spk_out_rec]
    

class LIF_ConDivNet2dStimSparse_countTimeReadout(torch.nn.Module):
    """spiking neural network in snntorch."""

    def __init__(self, timesteps, Nin, Nh, Nout, pcon, alpha):
        super().__init__()
        
        self.timesteps = timesteps # number of time steps to simulate the network
        self.inp_1 = int(Nin/4)
        self.inp_2 = int(Nin/4)
        self.inp_3 = int(Nin/4)
        self.inp_4 = int(Nin/4)
        self.h = Nh # number of hidden neurons 
        self.out = Nout
        self.p = pcon # connection sparsity (% of connections to set to zero)
        self.alpha = alpha
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function
        self.inp_1_offsets = 0.1*torch.randn(self.inp_1)
        self.inp_2_offsets = 0.1*torch.randn(self.inp_2)
        self.inp_3_offsets = 0.1*torch.randn(self.inp_3)
        self.inp_4_offsets = 0.1*torch.randn(self.inp_4)
        
        self.time_readout_vec = torch.nn.Parameter(0.1*torch.randn(Nout,2))
        self.count_readout_vec = torch.nn.Parameter(0.1*torch.randn(Nout,2))
#         print(f'self.readout_vec.shape = {self.readout_vec.shape}')
        
        # global parameters
        beta_min = 0.7
        beta_max = 0.9
        thr_min = 0.0
        thr_max = 0.5
        
##################################################################################################
        
        # stimulus to quadrant 1 input layer
        self.fc_in_1 = torch.nn.Linear(in_features=self.inp_1, out_features=self.inp_1, bias=False)
#         torch.nn.init.sparse_(self.fc_in_1.weight, sparsity=self.p, std=1./np.sqrt(1*self.p))
        # randomly initialize parameters for input layer
        beta_in_1 = (beta_max-beta_min)*torch.rand(self.inp_1) + beta_min
        thr_in_1 = torch.rand(self.inp_1)*(thr_max-thr_min) + thr_min
        self.lif_in_1 = snn.Leaky(beta=beta_in_1, 
                                threshold=thr_in_1, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 2 input layer
        self.fc_in_2 = torch.nn.Linear(in_features=self.inp_2, out_features=self.inp_2, bias=False)
#         torch.nn.init.sparse_(self.fc_in_2.weight, sparsity=self.p, std=1./np.sqrt(1*self.p))
        # randomly initialize parameters for input layer
        beta_in_2 = (beta_max-beta_min)*torch.rand(self.inp_1) + beta_min
        thr_in_2 = torch.rand(self.inp_2)*(thr_max-thr_min) + thr_min
        self.lif_in_2 = snn.Leaky(beta=beta_in_2, 
                                threshold=thr_in_2, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 3 input layer
        self.fc_in_3 = torch.nn.Linear(in_features=self.inp_3, out_features=self.inp_3, bias=False)
#         torch.nn.init.sparse_(self.fc_in_3.weight, sparsity=self.p, std=1.)
        # randomly initialize parameters for input layer
        beta_in_3 = (beta_max-beta_min)*torch.rand(self.inp_1) + beta_min
        thr_in_3 = torch.rand(self.inp_3)*(thr_max-thr_min) + thr_min
        self.lif_in_3 = snn.Leaky(beta=beta_in_3, 
                                threshold=thr_in_3, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # stimulus to quadrant 4 input layer
        self.fc_in_4 = torch.nn.Linear(in_features=self.inp_4, out_features=self.inp_4, bias=False)
#         torch.nn.init.sparse_(self.fc_in_4.weight, sparsity=self.p, std=1.)
        # randomly initialize parameters for input layer
        beta_in_4 = (beta_max-beta_min)*torch.rand(self.inp_1) + beta_min
        thr_in_4 = torch.rand(self.inp_4)*(thr_max-thr_min) + thr_min
        self.lif_in_4 = snn.Leaky(beta=beta_in_4, 
                                threshold=thr_in_4, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        
        # quadrant 1 input layer to E1 layer
        self.fc_h_1 = torch.nn.Linear(in_features=self.inp_1, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_1.weight, sparsity=self.p, std=1./np.sqrt(self.inp_1*(1-self.p)))
        # quadrant 2 input layer to E1 layer
        self.fc_h_2 = torch.nn.Linear(in_features=self.inp_2, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_2.weight, sparsity=self.p, std=1./np.sqrt(self.inp_2*(1-self.p)))
        # quadrant 1 input layer to E1 layer
        self.fc_h_3 = torch.nn.Linear(in_features=self.inp_3, out_features=self.h, bias=False)
        torch.nn.init.sparse_(self.fc_h_3.weight, sparsity=self.p, std=1./np.sqrt(self.inp_3*(1-self.p)))
        # quadrant 2 input layer to E1 layer
        self.fc_h_4 = torch.nn.Linear(in_features=self.inp_4, out_features=self.h, bias=False) 
        torch.nn.init.sparse_(self.fc_h_4.weight, sparsity=self.p, std=1./np.sqrt(self.inp_4*(1-self.p)))
        # randomly initialize parameters for E1 layer
        beta_h = (beta_max-beta_min)*torch.rand(self.h) + beta_min
        thr_h = torch.rand(self.h)*(thr_max-thr_min) + thr_min
        self.lif_h = snn.Leaky(beta=beta_h, threshold=thr_h, 
                                learn_threshold=False, spike_grad=spike_grad)
        
##################################################################################################

        # hidden layer to output layer
        self.fc_out = torch.nn.Linear(in_features=self.h, out_features=self.out,bias=False)
        torch.nn.init.sparse_(self.fc_out.weight, sparsity=self.p, std=1./np.sqrt(self.h*(1-self.p)))
        # randomly initialize parameters for B layer
        beta_out = (beta_max-beta_min)*torch.rand(self.out) + beta_min
        thr_out = torch.rand(self.out)*(thr_max-thr_min) + thr_min
        self.lif_out = snn.Leaky(beta=beta_out, threshold=thr_out, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # output layer to readout neuron
        self.fc_readout = torch.nn.Linear(in_features=self.out, out_features=2,bias=False)        
        # randomly initialize decay rate for single output neuron
        thr_readout = torch.ones(2)
        beta_readout = torch.rand(2)
        # leaky integrator readout neuron used for reconstruction
        self.li_readout = snn.Leaky(beta=beta_readout, threshold=thr_readout, learn_beta=True, 
                                    spike_grad=spike_grad, reset_mechanism="none")
        

    def forward(self, stim):
        """Forward pass for several time steps."""

        # Initalize membrane potential
#         pre_in_pos, post_in_pos, mem_in_pos = self.lif_in_pos.init_alpha()
#         pre_in_neg, post_in_neg, mem_in_neg = self.lif_in_neg.init_alpha()
        mem_in_1 = self.lif_in_1.init_leaky()
        mem_in_2 = self.lif_in_2.init_leaky()
        mem_in_3 = self.lif_in_3.init_leaky()
        mem_in_4 = self.lif_in_4.init_leaky()
        mem_h = self.lif_h.init_leaky()
        mem_out = self.lif_out.init_leaky()
        mem_readout = self.li_readout.init_leaky()

        # Empty lists to record outputs
        spk_in_1_rec = []
        spk_in_2_rec = []
        spk_in_3_rec = []
        spk_in_4_rec = []
        spk_h_rec = []
        spk_out_rec = []
        mem_readout_rec = []
#         readout_rec = []
        
        deltatt = 10
        kernelRange = torch.linspace(0,1,deltatt+1)
        sigma = 0.1
        half = kernelRange[int(deltatt/2)]
        k = torch.exp(-((kernelRange-half)**2)/sigma)
        time_kernel = torch.zeros((self.out,deltatt+1))
        for i in range(self.out):
            time_kernel[i,:] = k
            
        deltatc = 70
        kernelRange = torch.linspace(0,1,deltatc+1)
        sigma = 0.1
        half = kernelRange[int(deltatc/2)]
        k = torch.exp(-((kernelRange-half)**2)/sigma)
        count_kernel = torch.zeros((self.out,deltatc+1))
        for i in range(self.out):
            count_kernel[i,:] = k/(deltatc+1)
#         kernel = torch.tensor([torch.exp(-((x-half)**2)/sigma) for x in kernelRange])
#         plt.plot(kernel)
#         plt.show()
#         plt.close()
        
        # print(f'stim.shape = {stim.shape}')
        
        # loop over
        for step in range(self.timesteps):
            x = stim[step, 0, :]*torch.ones(self.inp_1)
            y = stim[step, 1, :]*torch.ones(self.inp_1)
            
            stim_1 = torch.sqrt(x[0]**2 + y[0]**2)*(x > self.inp_1_offsets)*(y > self.inp_1_offsets) # first quadrant
            cur_in_1 = self.fc_in_1(stim_1)
            spk_in_1, mem_in_1 = self.lif_in_1(cur_in_1,mem_in_1)
            
            stim_2 = torch.sqrt(x[0]**2 + y[0]**2)*(x < self.inp_2_offsets)*(y > self.inp_2_offsets) # second quadrant
            cur_in_2 = self.fc_in_2(stim_2)
            spk_in_2, mem_in_2 = self.lif_in_2(cur_in_2,mem_in_2)
            
            stim_3 = torch.sqrt(x[0]**2 + y[0]**2)*(x < self.inp_3_offsets)*(y < self.inp_3_offsets) # third quadrant
            cur_in_3 = self.fc_in_3(stim_3)
            spk_in_3, mem_in_3 = self.lif_in_3(cur_in_3,mem_in_3)
            
            stim_4 = torch.sqrt(x[0]**2 + y[0]**2)*(x > self.inp_4_offsets)*(y < self.inp_4_offsets) # fourth quadrant
            cur_in_4 = self.fc_in_4(stim_4)
            spk_in_4, mem_in_4 = self.lif_in_4(cur_in_4,mem_in_4)
            
            cur_h_1 = self.fc_h_1(spk_in_1)
            cur_h_2 = self.fc_h_2(spk_in_2)
            cur_h_3 = self.fc_h_3(spk_in_3)
            cur_h_4 = self.fc_h_4(spk_in_4)
            
            cur_h = cur_h_1 + cur_h_2 + cur_h_3 + cur_h_4
            spk_h, mem_h = self.lif_h(cur_h, mem_h)
            
            cur_out = self.fc_out(spk_h)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            
#             torchaudio.functional.convolve(spk_nrn0,kernel)
#             countVec = torch.tensor([torchaudio.functional.convolve(spk_out[:,i],kernel,mode='valid') for i in range(spk_out_rec.shape[1])])
            
#             print(f'spk_out.shape = {spk_out.shape}')
#             readout = torch.matmul(self.readout_vec,spk_out)
#             print(f'readout.shape = {readout.shape}')
#             cur_out = self.fc_out(spk_out)
            
            cur_readout = self.fc_readout(spk_out)
            _, mem_readout = self.li_readout(cur_readout, mem_readout)
            
#             spk_in_pos_rec.append(spk_in_pos)
#             spk_in_neg_rec.append(spk_in_neg)
            spk_in_1_rec.append(spk_in_1)
            spk_in_2_rec.append(spk_in_2)
            spk_in_3_rec.append(spk_in_3)
            spk_in_4_rec.append(spk_in_4)
            spk_h_rec.append(spk_h)
            spk_out_rec.append(spk_out)
            mem_readout_rec.append(mem_readout)
#             readout_rec.append(readout)
            
            
        mem_readout_rec = torch.stack(mem_readout_rec)
        mem_new = mem_readout_rec.view(self.timesteps, 2, 1)
#         readout_rec = torch.stack(readout_rec)
#         readout_new = readout_rec.view(self.timesteps, 2, 1)
#         spk_in_pos_rec = torch.stack(spk_in_pos_rec)
#         spk_in_neg_rec = torch.stack(spk_in_neg_rec)
        spk_in_1_rec = torch.stack(spk_in_1_rec)
        spk_in_2_rec = torch.stack(spk_in_2_rec)
        spk_in_3_rec = torch.stack(spk_in_3_rec)
        spk_in_4_rec = torch.stack(spk_in_4_rec)
        spk_h_rec = torch.stack(spk_h_rec)
        spk_out_rec = torch.stack(spk_out_rec)

        timevec = torchaudio.functional.convolve(torch.t(spk_out_rec),time_kernel,mode='valid')
        countvec = torchaudio.functional.convolve(torch.t(spk_out_rec),count_kernel,mode='valid')
        yhat_time_pre = torch.matmul(torch.t(timevec),self.time_readout_vec)
        yhat_count = torch.matmul(torch.t(countvec),self.count_readout_vec)
        tmax = yhat_count.shape[0]
        yhat_time = yhat_time_pre[:tmax,:]
#         print(f'yhat_count.shape = {yhat_count.shape}')
#         print(f'yhat_time.shape = {yhat_time.shape}')
        readout_rec = self.alpha*yhat_time + (1-self.alpha)*yhat_count
        
#         yhat_time = torch.reshape(yhat_time,yhat_count.shape)
#         yhat = self.alpha*yhat_time
        
#         num_times = countVec.shape[1]
#         readout_rec = torch.matmul(torch.t(countVec),self.readout_vec)
#         print(f'readout_rec.shape = {readout_rec.shape}')
        readout = readout_rec.view(tmax, 2, 1)
#         readout = 0
#         countVec = 
#         print(f'countVec.shape = {countVec.shape}')
        
        
        return [readout, mem_new, spk_in_1_rec, spk_in_2_rec, spk_in_3_rec, spk_in_4_rec, spk_h_rec, spk_out_rec]
    
    
    
def makeWtPlots(model_params,tag,seednum):
    if not os.path.exists('wtPlots'):
        os.mkdir('wtPlots')
    if not os.path.exists('wtPlots/' + f'{seednum}'):
        os.mkdir('wtPlots/' + f'{seednum}')
    path = 'wtPlots/' + f'{seednum}/' + tag
    if not os.path.exists(path):
        os.mkdir(path)
    
    in_wts_ = model_params[0]
    E1_wts_ = model_params[3]
    B_wts_ = model_params[5]
    E2_wts_ = model_params[7]
    out_wts_ = model_params[9]

    in_wts = in_wts_[1].detach().numpy()
    E1_wts = E1_wts_[1].detach().numpy()
    B_wts = B_wts_[1].detach().numpy()
    E2_wts = E2_wts_[1].detach().numpy()
    out_wts = out_wts_[1].detach().numpy()
    
    in_wt = in_wts.flatten()
    E1_wt = E1_wts.flatten()
    B_wt = B_wts.flatten()
    E2_wt = E2_wts.flatten()
    out_wt = out_wts.flatten()
    
    plt.hist(in_wt,density=True)
    plt.ylabel('Prob. density',fontsize=18)
    plt.xlabel('weight',fontsize=18)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    fname = 'in_wt.png'
    fullpath = path + '/' + fname
    plt.savefig(fullpath,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
    
    plt.hist(E1_wt,density=True)
    plt.ylabel('Prob. density',fontsize=18)
    plt.xlabel('weight',fontsize=18)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    fname = 'E1_wt.png'
    fullpath = path + '/' + fname
    plt.savefig(fullpath,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
    
    plt.hist(B_wt,density=True)
    plt.ylabel('Prob. density',fontsize=18)
    plt.xlabel('weight',fontsize=18)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    fname = 'B_wt.png'
    fullpath = path + '/' + fname
    plt.savefig(fullpath,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
    
    plt.hist(E2_wt,density=True)
    plt.ylabel('Prob. density',fontsize=18)
    plt.xlabel('weight',fontsize=18)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    fname = 'E2_wt.png'
    fullpath = path + '/' + fname
    plt.savefig(fullpath,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
    
    plt.hist(out_wt,density=True)
    plt.ylabel('Prob. density',fontsize=18)
    plt.xlabel('weight',fontsize=18)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    fname = 'out_wt.png'
    fullpath = path + '/' + fname
    plt.savefig(fullpath,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()


def makeRaster(Ns, stim, spikeTimes, spikeIndices, outVolts, seednum, tag):
#     spikeTimes = [stin, stE1, stB, stE2, stout]
#     spikeIndices = [siin, siE1, siB, siE2, siout]
    stin, stE1, stB, stE2, stout = spikeTimes
    siin, siE1, siB, siE2, siout = spikeIndices
#     Ns = [Nin, NE1, NB, NE2, Nout]
    Nin, NE1, NB, NE2, Nout = Ns
    
    if not os.path.exists('rasters'):
        os.mkdir('rasters')
    fig, ax = plt.subplots(7,figsize=(9,12),gridspec_kw={'height_ratios': [1, 2, 8, 2, 4, 2, 1]})

    for j in range(len(ax)):
    #     ax[j].set_xlim([0,num_steps-1])
        ax[j].tick_params(axis='y',labelsize=12)
        if j != len(ax)-1:
            ax[j].set_xticks([])
        else:
            ax[j].tick_params(axis='x',labelsize=15)
            ax[j].set_xlabel('time (ms)',fontsize=20)
    ax[0].plot(stim, color='black')
    # cur_in_rec_re
    ax[0].set_ylabel('s(t)',fontsize=15)
    ax[1].plot(stin,siin,marker='o',linestyle='',markersize=1, color='black')
    ax[1].set_ylabel('in',fontsize=18)
    ax[1].set_ylim([-1,Nin+1])
    ax[2].plot(stE1,siE1,marker='o',linestyle='',markersize=0.5, color='black')
    ax[2].set_ylabel('E1',fontsize=18)
    ax[2].set_ylim([-1,NE1+1])
    ax[3].plot(stB,siB,marker='o',linestyle='',markersize=1, color='black')
    ax[3].set_ylabel('B',fontsize=18)
    ax[3].set_ylim([-1,NB+1])
    ax[4].plot(stE2,siE2,marker='o',linestyle='',markersize=1, color='black')
    ax[4].set_ylim([-1,NE2+1])
    ax[4].set_ylabel('E2',fontsize=18)
    ax[5].plot(stout,siout,marker='o',linestyle='',markersize=1, color='black')
    ax[5].set_ylabel('out',fontsize=18)
    ax[5].set_ylim([-1,Nout+1])
    ax[6].set_ylabel('volts',fontsize=15)
    ax[6].plot(outVolts, color='black')
    fname = tag + f'{seednum}.png'
    path = 'rasters/' + fname
    plt.savefig(path,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
    
def get2DcorrPlots(st,si,stim,N_nrn,tf,T_R,dt,delta_t,seednum,delay,layerName,Nh,tag,saveData=False):
    timeVec, countVec, rankVec = getSlidingVecsLog(st,si,N_nrn,tf,T_R,dt,delta_t,seednum,delay)
    
    dataDir = 'dataDir'
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
        
    seedDir = f'seed{seednum}'
    # if not os.path.exists(dataDir + '/' + seedDir):
    #     os.mkdir(dataDir + '/' + seedDir)
    
    # dataPath = dataDir + '/' + seedDir + f'/Nh{Nh}'
    # if not os.path.exists(dataPath):
    #     os.mkdir(dataPath)

    if saveData:
        filename = dataPath + '/' + 'times_'+ layerName + '_' + tag + '.txt'
        file = open(filename,'w+')
        np.savetxt(file,timeVec)
        file.close()

        filename = dataPath + '/' + 'counts_'+ layerName + '_' + tag + '.txt'
        file = open(filename,'w+')
        np.savetxt(file,countVec)
        file.close()

        filename = dataPath + '/' + 'ranks_'+ layerName + '_' + tag + '.txt'
        file = open(filename,'w+')
        np.savetxt(file,rankVec)
        file.close()    
        
    plotDir = 'corrPlots'
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
        
    if not os.path.exists(plotDir + '/' + seedDir):
        os.mkdir(plotDir + '/' + seedDir)
        
    plotPath = plotDir + '/' + seedDir + '/' + f'/Nh{Nh}'
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
        
    plotPath = plotDir + '/' + seedDir + '/' + f'/Nh{Nh}' + tag
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
        
    fnameCount = plotPath + '/predictedVtrue_'+ layerName + '_count.png'
    fnameRank = plotPath + '/predictedVtrue_'+ layerName + '_rank.png'
    fnameTime = plotPath + '/predictedVtrue_'+ layerName + '_time.png'
    
    y = stim
    XT = timeVec.T
    XC = countVec.T
    XR = rankVec.T
    
    #################### TIMING CROSS-VALIDATION, BAYESIAN OPTIMIZATION VERSION ######################
    XT_train, XT_validate, y_train, y_validate = train_test_split(XT, y, train_size=0.5,shuffle=False)
    # print(f'y_train.shape = {y_train.shape}')
    XT_validate, XT_test, y_validate, y_test = train_test_split(XT_validate, y_validate, test_size=0.5,shuffle=False)
    xpos = y_test[:,0]
    ypos = y_test[:,1]
    
    def svr(c):
        svr_rbf_multi = MultiOutputRegressor(SVR(kernel="rbf", C=c))
        svr_rbf_multi.fit(XT_train,y_train)
        r2_train = svr_rbf_multi.score(XT_train,y_train)
        r2_valid = svr_rbf_multi.score(XT_validate,y_validate)
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
    svr_rbf_multi.fit(XT_train,y_train)
    y_pred_time = svr_rbf_multi.predict(XT_test)

#     print(f'y_pred_time.shape = {y_pred_time.shape}')
    xposhat = y_pred_time[:,0]
    yposhat = y_pred_time[:,1]

    r2_train_time = svr_rbf_multi.score(XT_train,y_train)
    r2_test_time = svr_rbf_multi.score(XT_test,y_test)
    ####################################################################
    fig, ax = plt.subplots(1,2,figsize=(6,3))
    ax[0].plot(xposhat,xpos,marker='o', linestyle='', color='gray')
    m, b = np.polyfit(xposhat, xpos, 1)
    ax[0].plot(xposhat, m*xposhat+b,color = 'blue',linewidth=3)
    ax[0].set_xlabel(r'$\hat{s}_x$',fontsize=20)
    ax[0].set_ylabel(r'$s_x$',fontsize=20)
    # ax[1].set_yticks([])
    ax[1].plot(yposhat,ypos,marker='o', linestyle='', color='gray')
    m, b = np.polyfit(yposhat, ypos, 1)
    ax[1].plot(yposhat, m*yposhat+b,color = 'blue',linewidth=3)
    ax[1].set_ylabel(r'$s_y$',fontsize=20)
    ax[1].set_xlabel(r'$\hat{s}_y$',fontsize=20)
    plt.tight_layout()
    plt.savefig(fnameTime,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
    
    
    #################### COUNT CROSS-VALIDATION, BAYESIAN OPTIMIZATION VERSION ######################
    XC_train, XC_validate, y_train, y_validate = train_test_split(XC, y, train_size=0.5,shuffle=False)
    XC_validate, XC_test, y_validate, y_test = train_test_split(XC_validate, y_validate, test_size=0.5,shuffle=False)
    
    def svr(c):
        svr_rbf_multi = MultiOutputRegressor(SVR(kernel="rbf", C=c))
        svr_rbf_multi.fit(XC_train,y_train)
        r2_train = svr_rbf_multi.score(XC_train,y_train)
        r2_valid = svr_rbf_multi.score(XC_validate,y_validate)
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
    svr_rbf_multi.fit(XC_train,y_train)
    y_pred_count = svr_rbf_multi.predict(XC_test)

    xposhat = y_pred_count[:,0]
    yposhat = y_pred_count[:,1]

    r2_train_count = svr_rbf_multi.score(XC_train,y_train)
    r2_test_count = svr_rbf_multi.score(XC_test,y_test)
    ####################################################################
    fig, ax = plt.subplots(1,2,figsize=(6,3))
    ax[0].plot(xposhat,xpos,marker='o', linestyle='', color='gray')
    m, b = np.polyfit(xposhat, xpos, 1)
    ax[0].plot(xposhat, m*xposhat+b,color = 'magenta',linewidth=3)
    ax[0].set_xlabel(r'$\hat{s}$',fontsize=20)
    ax[0].set_ylabel(r'$s$',fontsize=20)
    ax[1].set_yticks([])
    ax[1].plot(yposhat,ypos,marker='o', linestyle='', color='gray')
    m, b = np.polyfit(yposhat, ypos, 1)
    ax[1].plot(yposhat, m*yposhat+b,color = 'magenta',linewidth=3)
    ax[1].set_xlabel(r'$\hat{s}$',fontsize=20)
    plt.savefig(fnameCount,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
    
    
    #################### RANK CROSS-VALIDATION, BAYESIAN OPTIMIZATION VERSION ######################
    XR_train, XR_validate, y_train, y_validate = train_test_split(XR, y, train_size=0.5,shuffle=False)
    XR_validate, XR_test, y_validate, y_test = train_test_split(XR_validate, y_validate, test_size=0.5,shuffle=False)
    
    def svr(c):
        svr_rbf_multi = MultiOutputRegressor(SVR(kernel="rbf", C=c))
        svr_rbf_multi.fit(XR_train,y_train)
        r2_train = svr_rbf_multi.score(XR_train,y_train)
        r2_valid = svr_rbf_multi.score(XR_validate,y_validate)
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
    svr_rbf_multi.fit(XR_train,y_train)
    y_pred_rank = svr_rbf_multi.predict(XR_test)

    xposhat = y_pred_rank[:,0]
    yposhat = y_pred_rank[:,1]

    r2_train_rank = svr_rbf_multi.score(XR_train,y_train)
    r2_test_rank = svr_rbf_multi.score(XR_test,y_test)
    ####################################################################
    fig, ax = plt.subplots(1,2,figsize=(6,3))
    ax[0].plot(xposhat,xpos,marker='o', linestyle='', color='gray')
    m, b = np.polyfit(xposhat, xpos, 1)
    ax[0].plot(xposhat, m*xposhat+b,color = 'orange',linewidth=3)
    ax[0].set_xlabel(r'$\hat{s}$',fontsize=20)
    ax[0].set_ylabel(r'$s$',fontsize=20)
    ax[1].set_yticks([])
    ax[1].plot(yposhat,ypos,marker='o', linestyle='', color='gray')
    m, b = np.polyfit(yposhat, ypos, 1)
    ax[1].plot(yposhat, m*yposhat+b,color = 'orange',linewidth=3)
    ax[1].set_xlabel(r'$\hat{s}$',fontsize=20)
    plt.savefig(fnameRank,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
    
    fitMetrics = [r2_train_count, r2_test_count, r2_train_rank, r2_test_rank, r2_train_time, r2_test_time]
    data = [y_test, y_pred_count, y_pred_rank, y_pred_time]
    return fitMetrics, data

def plot2Dreconstructions(data,layerName,Nh,seednum,tag,showRank=False):
    plotDir = 'reconstructions'
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
        
    seedDir = f'{seednum}'
    if not os.path.exists(plotDir + '/' + seedDir):
        os.mkdir(plotDir + '/' + seedDir)    
    
    plotPath = plotDir + '/' + seedDir + '/' + f'/Nh{Nh}'
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
        
    fname = plotPath + '/reconstruction_'+ layerName + f'_{tag}.png'
    
    y_test, y_pred_count, y_pred_rank, y_pred_time = data

    x_true = y_test[:,0]
    y_true = y_test[:,1]
    xhat_count = y_pred_count[:,0]
    yhat_count = y_pred_count[:,1]
    xhat_time = y_pred_time[:,0]
    yhat_time = y_pred_time[:,1]
    xhat_rank = y_pred_rank[:,0]
    yhat_rank = y_pred_rank[:,1]        

    if showRank:
        fig,ax = plt.subplots(1,3,figsize=(9,3))
        for axi in range(len(ax)):
            ax[axi].set_xlim([-1,1])
            ax[axi].set_ylim([-1,1])
        ax[0].plot(xhat_count,yhat_count,color='magenta',alpha=0.5,linewidth=3)
        ax[0].plot(x_true,y_true,color='black',alpha=0.5,linewidth=3)
        ax[0].set_xlabel('x',fontsize=20)
        ax[0].set_ylabel('y',fontsize=20)
        ax[1].plot(xhat_time,yhat_time,color='blue',alpha=0.5,linewidth=3)
        ax[1].plot(x_true,y_true,color='black',alpha=0.5,linewidth=3)
        ax[1].set_xlabel('x',fontsize=20)
        ax[2].plot(xhat_rank,yhat_rank,color='orange',alpha=0.5,linewidth=3)
        ax[2].plot(x_true,y_true,color='black',alpha=0.5,linewidth=3)
        ax[2].set_xlabel('x',fontsize=20)
        plt.tight_layout()
    #     plt.show()
        plt.savefig(fname,bbox_inches='tight',dpi=200)
        plt.close()

    else:
        fig,ax = plt.subplots(1,2,figsize=(6,3))
        for axi in range(len(ax)):
            ax[axi].set_xlim([-1,1])
            ax[axi].set_ylim([-1,1])
        ax[0].plot(xhat_count,yhat_count,color='magenta',alpha=0.5,linewidth=3)
        ax[0].plot(x_true,y_true,color='black',alpha=0.5,linewidth=3)
        ax[0].set_xlabel('x',fontsize=20)
        ax[0].set_ylabel('y',fontsize=20)
        ax[1].plot(xhat_time,yhat_time,color='blue',alpha=0.5,linewidth=3)
        ax[1].plot(x_true,y_true,color='black',alpha=0.5,linewidth=3)
        ax[1].set_xlabel('x',fontsize=20)
        plt.tight_layout()
    #     plt.show()
        plt.savefig(fname,bbox_inches='tight',dpi=200)
        plt.close()


# getReconstruction(tvecSub,y_test,y_pred_count,y_pred_rank,layerName,tag,seednum)
def getReconstruction(tvecSub,y_test,y_pred_count,y_pred_rank,y_pred_time,layerName,Nh,seednum):
    plotDir = 'reconstructions'
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
        
    seedDir = f'{seednum}'
    if not os.path.exists(plotDir + '/' + seedDir):
        os.mkdir(plotDir + '/' + seedDir)    
    
    plotPath = plotDir + '/' + seedDir + '/' + f'/Nh{Nh}'
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
        
    fname = plotPath + '/reconstruction_'+ layerName + '.png'
    
    N_test = len(y_test)
    fig, ax = plt.subplots(3,figsize=(9,9))
    for axi in ax:
        axi.tick_params(axis='x',labelsize=12)
        axi.tick_params(axis='y',labelsize=12)
    t_train = tvecSub[:-N_test]
    t_test = tvecSub[-N_test:]
    t_start = t_test[0]
    t_end = t_test[-1]
    ax[0].set_title('reconstructions',fontsize=25)
    ax[0].plot(t_test,y_pred_count,marker='',linestyle='-',color='magenta',linewidth=2,alpha=0.8)
    ax[0].set_xlim([t_start,t_end])
    # ax[0].set_ylim([-2,2])
    ax[0].plot(t_test,y_test,marker='',linestyle='-',color='black',label='true',linewidth=3,alpha=0.8)
    ax[0].set_ylabel('stimulus',fontsize=20)
    black_patch = mpatches.Patch(color='black', label='true')
    magenta_patch = mpatches.Patch(color='magenta', label='count')
    orange_patch = mpatches.Patch(color='orange', label='rank')
    blue_patch = mpatches.Patch(color='blue', label='time')
    ax[1].plot(t_test,y_pred_rank,marker='',linestyle='-',color='orange',linewidth=1,alpha=0.8)
    # ax[1].set_ylim([-2,2])
    ax[1].plot(t_test,y_test,marker='',linestyle='-',color='black',label='true',linewidth=3,alpha=0.8)
    ax[1].set_ylabel('stimulus',fontsize=20)
#     ax[1].legend(handles=[black_patch, magenta_patch, orange_patch],bbox_to_anchor=(1.25,1.4),fontsize=15)
    ax[1].set_xlim([t_start,t_end])
#     ax[1].set_xlabel(r'time (ms)',fontsize=20)
    ax[2].plot(t_test,y_pred_time,marker='',linestyle='-',color='blue',linewidth=1,alpha=0.8)
    ax[2].plot(t_test,y_test,marker='',linestyle='-',color='black',label='true',linewidth=3,alpha=0.8)
    ax[2].set_ylabel('stimulus',fontsize=20)
    ax[2].legend(handles=[black_patch, magenta_patch, orange_patch, blue_patch],bbox_to_anchor=(1.25,1.7),fontsize=15)
    ax[2].set_xlim([t_start,t_end])
    ax[2].set_xlabel(r'time (ms)',fontsize=20)
#     fname = 'reconstruct_'+ layerName +'_' + tag + f'_{seednum}.png'
#     path = dirName + '/' + fname
    plt.savefig(fname,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
# getStimDist(y_test,y_pred_count,y_pred_rank,layerName,tag,seednum)
def getStimDist(y_test,y_pred_count,y_pred_rank,y_pred_time,layerName,Nh,seednum):
    plotDir = 'stimDists'
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
        
    seedDir = f'{seednum}'
    if not os.path.exists(plotDir + '/' + seedDir):
        os.mkdir(plotDir + '/' + seedDir)
    
    plotPath = plotDir + '/' + seedDir + '/' + f'/Nh{Nh}'
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
        
    fname = plotPath + '/stimDist_'+ layerName + '.png'
    
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    for axi in ax:
        axi.tick_params(axis='x',labelsize=12)
        axi.tick_params(axis='y',labelsize=12)
    gray_patch = mpatches.Patch(color='gray', label='true')
    magenta_patch = mpatches.Patch(color='magenta', label='count')
    orange_patch = mpatches.Patch(color='orange', label='rank')
    blue_patch = mpatches.Patch(color='blue', label='time')
    x_true = ax[0].hist(y_test,density=True,bins=25,alpha=0.7,color='gray')
    ax[1].hist(y_test,density=True,bins=25,alpha=0.7,color='gray',label='true')
    ax[2].hist(y_test,density=True,bins=25,alpha=0.7,color='gray',label='true')
    # true stimulus
    px_true = x_true[0]
    dx_true = np.diff(x_true[1])
    p_true = px_true*dx_true
    true_ent = np.nansum(-p_true*np.log2(p_true))
#     print(true_ent)
    # count reconstruction
    x_pred_count = ax[0].hist(y_pred_count,density=True,bins=25,alpha=0.7,color='magenta')
    px_count = x_pred_count[0]
    dx_count = np.diff(x_pred_count[1])
    p_count = px_count*dx_count
    count_ent = np.nansum(-p_count[p_count>0]*np.log2(p_count[p_count>0]))
    # rank reconstruction
    x_pred_rank = ax[1].hist(y_pred_rank,density=True,bins=25,alpha=0.7,color='orange')
    px_rank = x_pred_rank[0]
    dx_rank = np.diff(x_pred_rank[1])
    p_rank = px_rank*dx_rank
    rank_ent = np.nansum(-p_rank[p_rank>0]*np.log2(p_rank[p_rank>0]))
    # time reconstruction
    x_pred_time = ax[2].hist(y_pred_time,density=True,bins=25,alpha=0.7,color='blue')
    px_time = x_pred_time[0]
    dx_time = np.diff(x_pred_time[1])
    p_time = px_time*dx_time
    time_ent = np.nansum(-p_time[p_time>0]*np.log2(p_time[p_time>0]))  
    ax[0].set_title(f'entropy = {count_ent:.4} bits',fontsize=25,color='magenta')
    ax[1].set_title(f'entropy = {rank_ent:.4} bits',fontsize=25,color='orange')
    ax[2].set_title(f'entropy = {time_ent:.4} bits',fontsize=25,color='blue')
    ax[0].set_ylabel('probability density',fontsize=20)
    ax[2].set_xlabel('stimulus',fontsize=20)
    ax[1].set_xlabel('stimulus',fontsize=20)
    ax[0].set_xlabel('stimulus',fontsize=20)
    ax[2].legend(handles=[gray_patch, magenta_patch, orange_patch],bbox_to_anchor=(1.5,0.75),fontsize=15)
#     fname = 'stimDist_'+ layerName +'_' + tag + '.png'
#     path = dirName + '/' + fname
    plt.savefig(fname,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
    
    return [true_ent, count_ent, rank_ent, time_ent]

def plotFit(mem, label, seednum, Nh):
    if not os.path.exists('misc'):
        os.mkdir('misc')
    seedDir = f'seed{seednum}'
    if not os.path.exists('misc/' + seedDir):
        os.mkdir('misc/' + seedDir)
        
    plt.plot(mem[:, 0, 0], label="Output")
    plt.plot(label[:, 0, 0], '--', label="Target")
    plt.title(f'Nh = {Nh}',fontsize=18)
    plt.xlabel("time",fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("membrane potential",fontsize=18)
    plt.legend(loc='best',fontsize=15)
    
    fname = f'Nh{Nh}.png'
    path = 'misc/' + seedDir + fname
    plt.savefig(path,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()


def makeRaster2D(s, spk1, spk2, spk3, spk4, spkh, spkout, mem, Nh, seednum, tag):
    if not os.path.exists('rasters'):
        os.mkdir('rasters')
    if not os.path.exists(f'rasters/Nh{Nh}'):
        os.mkdir(f'rasters/Nh{Nh}')
    fname = f'rasters/Nh{Nh}/seed{seednum}_{tag}.png'

    xpos = s[:,0, 0].detach().numpy()
    ypos = s[:,1, 0].detach().numpy()
    memx = mem[:,0,0].detach().numpy()
    memy = mem[:,1,0].detach().numpy()
    spk_1_plot = spk1.detach().numpy().T
    spk_2_plot = spk2.detach().numpy().T
    spk_3_plot = spk3.detach().numpy().T
    spk_4_plot = spk4.detach().numpy().T
    spk_in_plot = np.vstack((spk_1_plot,spk_2_plot,spk_3_plot,spk_4_plot))
    spk_h_plot = spkh.detach().numpy().T
    spk_out_plot = spkout.detach().numpy().T
    # mem_plot = mem.detach().numpy()[:,0]
    Nin = spk_in_plot.shape[0]
    Nout = spk_out_plot.shape[0]
    num_steps = len(xpos)
    
    fig, ax = plt.subplots(5,figsize=(5,7),gridspec_kw={'height_ratios': [1, 3, 3, 3, 1]})
    for axi in range(len(ax)):
        ax[axi].set_xlim([0,num_steps])
        if axi != len(ax) - 1:
            ax[axi].set_xticks([])
    ax[0].plot(xpos,color='blue',label='x')
    ax[0].plot(ypos,color='red',label='y')
    ax[0].set_ylabel('stim')
    ax[0].legend(ncol=2,bbox_to_anchor=(0.3,1))
    ax[1].set_ylim([-1,Nin])
    ax[1].set_yticks([0,Nin])
    ax[1].imshow(spk_in_plot, cmap="binary",aspect="auto")
    ax[1].set_ylabel('input')
    ax[2].set_ylim([-1,Nh])
    ax[2].set_yticks([0,Nh])
    ax[2].imshow(spk_h_plot, cmap="binary",aspect="auto")
    ax[2].set_ylabel('hidden')
    ax[3].set_ylim([-1,Nout])
    ax[3].set_yticks([0,Nout])
    ax[3].imshow(spk_out_plot, cmap="binary",aspect="auto")
    ax[3].set_ylabel('output')
    ax[3].set_xticks([])
    ax[4].plot(memx,color='blue')
    ax[4].plot(memy,color='red')
    ax[4].set_ylabel('readout')
    ax[4].set_xlabel('time',fontsize=20)
    plt.savefig(fname,bbox_inches='tight',dpi=200)
    # plt.show()
    plt.close()


def plot2Dfit(s,mem,seednum,Nh,tag):
    if not os.path.exists('fitPlots'):
        os.mkdir('fitPlots')
    if not os.path.exists(f'fitPlots/Nh{Nh}'):
        os.mkdir(f'fitPlots/Nh{Nh}')
    fname = f'fitPlots/Nh{Nh}/seed{seednum}_{tag}.png'

    xpos = s[:,0, 0].detach().numpy()
    ypos = s[:,1, 0].detach().numpy()
    memx = mem[:,0,0].detach().numpy()
    memy = mem[:,1,0].detach().numpy()

    fig, ax = plt.subplots(2)
    ax[0].plot(xpos,color='black')
    ax[0].plot(memx,color='blue')
    ax[0].set_ylim([-1,1])
    ax[0].set_xticks([])
    ax[0].set_ylabel(r'$s_x$',fontsize=20)
    ax[1].set_ylim([-1,1])
    ax[1].plot(ypos,color='black')
    ax[1].plot(memy,color='red')
    ax[1].set_ylabel(r'$s_y$',fontsize=20)
    ax[1].set_xlabel('time',fontsize=20)
    plt.savefig(fname,bbox_inches='tight',dpi=200)
    plt.close()
