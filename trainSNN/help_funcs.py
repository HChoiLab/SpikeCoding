# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

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

from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import os
from bayes_opt import BayesianOptimization


def getRank(st,delta_t,t1,T_R):
    st_mod = st - t1
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
            sub_st = sub_st[sub_st < t2]
            countVec[ni][ti] = len(sub_st)
            rankVec[ni][ti] = getRank(st,delta_t,t1,T_R)
            ti+=1
    
    return [countVec, rankVec]


class RegressionDataset(torch.utils.data.Dataset):
    """Simple regression dataset."""

    def __init__(self, timesteps, num_samples, mode):
        """Linear relation between input and output"""
        self.num_samples = num_samples # number of generated samples
        feature_lst = [] # store each generated sample in a list

        # generate linear functions one by one
        for idx in range(num_samples):
#             end = float(torch.rand(1)) # random final point
            end = float(1)
            lin_vec = torch.linspace(start=0.0, end=end, steps=timesteps) # generate linear function from 0 to end
            feature = lin_vec.view(timesteps, 1)
            feature_lst.append(feature) # add sample to list

        self.features = torch.stack(feature_lst, dim=1) # convert list to tensor

        # option to generate linear function or square-root function
        if mode == "linear":
            self.labels = self.features * 1

        elif mode == "sqrt":
            slope = float(torch.rand(1))
            self.labels = torch.sqrt(self.features * slope)
            
        elif mode == 'sine':
#             print(self.features)
            self.labels = 0.9*torch.sin(2.*torch.pi*5.*self.features) + 1

        else:
            raise NotImplementedError("'linear', 'sqrt', 'sine'")

    def __len__(self):
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """General implementation, but we only have one sample."""
        return self.features[:, idx, :], self.labels[:, idx, :]
    
    
class ConDivNet(torch.nn.Module):
    """spiking neural network in snntorch."""

    def __init__(self, timesteps, Nin, NE1, NB, NE2, Nout):
        super().__init__()
        
        self.timesteps = timesteps # number of time steps to simulate the network
#         print(self.timesteps)
        self.inp = Nin
        self.E1 = NE1 # number of E1 neurons 
        self.B = NB
        self.E2 = NE2
        self.out = Nout
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function
        
        # global parameters
        alpha_min = 0.7
        alpha_max = 0.9
        
        # randomly initialize threshold for input layer
        alpha_in = (alpha_max-alpha_min)*torch.rand(self.inp) + alpha_min
        beta_in = alpha_in - 0.1
        thr_in = torch.rand(self.inp)

        # input layer
        self.fc_in = torch.nn.Linear(in_features=1, out_features=self.inp)
        self.lif_in = snn.Alpha(alpha=alpha_in, beta=beta_in, 
                                threshold=thr_in, 
                                learn_threshold=True, spike_grad=spike_grad)
        
        # randomly initialize threshold for E1 layer
        alpha_E1 = (alpha_max-alpha_min)*torch.rand(self.E1) + alpha_min
        beta_E1 = alpha_E1 - 0.1
        thr_E1 = torch.rand(self.E1)

        # E1 layer
        self.fc_E1 = torch.nn.Linear(in_features=self.inp, out_features=self.E1)
        self.lif_E1 = snn.Alpha(alpha=alpha_E1, beta=beta_E1, threshold=thr_E1, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # randomly initialize threshold for B layer
        alpha_B = (alpha_max-alpha_min)*torch.rand(self.B) + alpha_min
        beta_B = alpha_B - 0.1
        thr_B = 0.6*torch.rand(self.B)

        # B layer
        self.fc_B = torch.nn.Linear(in_features=self.E1, out_features=self.B)
        self.lif_B = snn.Alpha(alpha=alpha_B, beta=beta_B, threshold=thr_B, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # randomly initialize threshold for E2 layer
        alpha_E2 = (alpha_max-alpha_min)*torch.rand(self.E2) + alpha_min
        beta_E2 = alpha_E2 - 0.1
        thr_E2 = torch.rand(self.E2)

        # E2 layer
        self.fc_E2 = torch.nn.Linear(in_features=self.B, out_features=self.E2)
        self.lif_E2 = snn.Alpha(alpha=alpha_E2, beta=beta_E2, threshold=thr_E2, 
                                learn_threshold=False, spike_grad=spike_grad)
        
        # randomly initialize threshold for input layer
        alpha_out = (alpha_max-alpha_min)*torch.rand(self.out) + alpha_min
        beta_out = alpha_out - 0.1
        thr_out = torch.rand(self.out)

        # output layer
        self.fc_out = torch.nn.Linear(in_features=self.E2, out_features=self.out)
        self.lif_out = snn.Alpha(alpha=alpha_out, beta=beta_out, 
                                threshold=thr_out, 
                                learn_threshold=True, spike_grad=spike_grad)
        
        # randomly initialize decay rate for output neuron
        beta_readout = torch.rand(1)
        
        # leaky integrator readout neuron used for reconstruction
        self.fc_readout = torch.nn.Linear(in_features=self.out, out_features=1)
        self.li_readout = snn.Leaky(beta=beta_readout, threshold=1.0, learn_beta=True, 
                                    spike_grad=spike_grad, reset_mechanism="none")

    def forward(self, x):
        """Forward pass for several time steps."""

        # Initalize membrane potential
        presyn_in, postsyn_in, mem_in = self.lif_in.init_alpha()
        presyn_E1, postsyn_E1, mem_E1 = self.lif_E1.init_alpha()
        presyn_B, postsyn_B, mem_B = self.lif_B.init_alpha()
        presyn_E2, postsyn_E2, mem_E2 = self.lif_E2.init_alpha()
        presyn_out, postsyn_out, mem_out = self.lif_out.init_alpha()
        mem_readout = self.li_readout.init_leaky()

        # Empty lists to record outputs
        spk_in_rec = []
        spk_E1_rec = []
        spk_B_rec = []
        spk_E2_rec = []
        spk_out_rec = []
        mem_readout_rec = []
        
        # loop over
        for step in range(self.timesteps):
            stimulus = x[step, :, :]
            
            cur_in = stimulus*torch.ones(self.inp)
            spk_in, presyn_in, postsyn_in, mem_in = self.lif_in(cur_in, presyn_in, postsyn_in, mem_in)
            
            cur_E1 = self.fc_E1(spk_in)
            spk_E1, presyn_E1, postsyn_E1, mem_E1 = self.lif_E1(cur_E1, presyn_E1, postsyn_E1, mem_E1)
            
            cur_B = self.fc_B(spk_E1)
            spk_B, presyn_B, postsyn_B, mem_B = self.lif_B(cur_B, presyn_B, postsyn_B, mem_B)
            
            cur_E2 = self.fc_E2(spk_B)
            spk_E2, presyn_E2, postsyn_E2, mem_E2 = self.lif_E2(cur_E2, presyn_E2, postsyn_E2, mem_E2)
            
            cur_out = self.fc_out(spk_E2)
            spk_out, presyn_out, postsyn_out, mem_out = self.lif_out(cur_out, presyn_out, postsyn_out, mem_out)
            
            cur_readout = self.fc_readout(spk_out)
            _, mem_readout = self.li_readout(cur_readout, mem_readout)
            
            spk_in_rec.append(spk_in)
            spk_E1_rec.append(spk_E1)
            spk_B_rec.append(spk_B)
            spk_E2_rec.append(spk_E2)
            spk_out_rec.append(spk_out)
            mem_readout_rec.append(mem_readout)
            
        mem_readout_rec = torch.stack(mem_readout_rec)
        spk_in_rec = torch.stack(spk_in_rec)
        spk_E1_rec = torch.stack(spk_E1_rec)
        spk_B_rec = torch.stack(spk_B_rec)
        spk_E2_rec = torch.stack(spk_E2_rec)
        spk_out_rec = torch.stack(spk_out_rec)
            
        return [mem_readout_rec, spk_in_rec, spk_E1_rec, spk_B_rec, spk_E2_rec, spk_out_rec]
    
    
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
    ax[1].plot(stin,siin,marker='o',linestyle='',markersize=2, color='black')
    ax[1].set_ylabel('in',fontsize=18)
    ax[1].set_ylim([-1,Nin+1])
    ax[2].plot(stE1,siE1,marker='o',linestyle='',markersize=0.5, color='black')
    ax[2].set_ylabel('E1',fontsize=18)
    ax[2].set_ylim([-1,NE1+1])
    ax[3].plot(stB,siB,marker='o',linestyle='',markersize=2, color='black')
    ax[3].set_ylabel('B',fontsize=18)
    ax[3].set_ylim([-1,NB+1])
    ax[4].plot(stE2,siE2,marker='o',linestyle='',markersize=1, color='black')
    ax[4].set_ylim([-1,NE2+1])
    ax[4].set_ylabel('E2',fontsize=18)
    ax[5].plot(stout,siout,marker='o',linestyle='',markersize=2, color='black')
    ax[5].set_ylabel('out',fontsize=18)
    ax[5].set_ylim([-1,Nout+1])
    ax[6].set_ylabel('volts',fontsize=15)
    ax[6].plot(outVolts, color='black')
    fname = tag + f'{seednum}.png'
    path = 'rasters/' + fname
    plt.savefig(path,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()

def getCorrPlots(st,si,stim,N_nrn,tf,T_R,dt,delta_t,seednum,delay,layerName,tag):
    print('\tgetting count and rankvecs...')
    countVec, rankVec = getSlidingCountRankVecs(st,si,N_nrn,tf,T_R,dt,delta_t,seednum,delay)
    
    dataDir = 'dataDir'
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
        
    seedDir = f'{seednum}'
    if not os.path.exists(dataDir + '/' + seedDir):
        os.mkdir(dataDir + '/' + seedDir)
    
    dataPath = dataDir + '/' + seedDir + '/' + tag
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)

    filename = dataPath + '/' + 'counts_'+ layerName + '.txt'
    file = open(filename,'w+')
    np.savetxt(file,countVec)
    file.close()

    filename = dataPath + '/' + 'ranks_'+ layerName + '.txt'
    file = open(filename,'w+')
    np.savetxt(file,rankVec)
    file.close()    
        
    plotDir = 'corrPlots'
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
        
    if not os.path.exists(plotDir + '/' + seedDir):
        os.mkdir(plotDir + '/' + seedDir)
        
    plotPath = plotDir + '/' + seedDir + '/' + tag
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
        
    fname = plotPath + '/predictedVtrue_'+ layerName + '.png'
    
    y = stim
    XC = countVec.T
    XR = rankVec.T
    
    print('\tcount optimization ...')
    #################### count CROSS-VALIDATION, BAYESIAN OPTIMIZATION VERSION ######################
    XC_train, XC_validate, y_train, y_validate = train_test_split(XC, y, train_size=0.5,shuffle=False)
    XC_validate, XC_test, y_validate, y_test = train_test_split(XC_validate, y_validate, test_size=0.5,shuffle=False)
    def svr(c):
        svr_rbf = SVR(kernel="rbf", C=c)
        svr_rbf.fit(XC_train,y_train)
        r2_train = svr_rbf.score(XC_train,y_train)
        r2_valid = svr_rbf.score(XC_validate,y_validate)
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
    svr_rbf = SVR(kernel="rbf", C=bestC)
    svr_rbf.fit(XC_train,y_train)
    y_pred_count = svr_rbf.predict(XC_test)
    r2_train_count = svr_rbf.score(XC_train,y_train)
    r2_test_count = svr_rbf.score(XC_test,y_test)
    ####################################################################
    m_count, b_count = np.polyfit(y_test, y_pred_count, 1)
    x = y_test

    print('\trank optimization')
    #################### RANK CROSS-VALIDATION, BAYESIAN OPTIMIZATION VERSION ######################
    XR_train, XR_validate, y_train, y_validate = train_test_split(XR, y, train_size=0.5,shuffle=False)
    XR_validate, XR_test, y_validate, y_test = train_test_split(XR_validate, y_validate, test_size=0.5,shuffle=False)
    def svr(c):
        svr_rbf = SVR(kernel="rbf", C=c)
        svr_rbf.fit(XR_train,y_train)
        r2_train = svr_rbf.score(XR_train,y_train)
        r2_valid = svr_rbf.score(XR_validate,y_validate)
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
    svr_rbf = SVR(kernel="rbf", C=bestC)
    svr_rbf.fit(XR_train,y_train)
    y_pred_rank = svr_rbf.predict(XR_test)
    r2_train_rank = svr_rbf.score(XR_train,y_train)
    r2_test_rank = svr_rbf.score(XR_test,y_test)
    ####################################################################
    m_rank, b_rank = np.polyfit(y_test, y_pred_rank, 1)

    ############# PLOTTING ##################
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    x = y_test
    ax[0].set_title(f'count ($R^2={r2_test_count:.3}$)',fontsize=25)
    ax[0].plot(x, m_count*x+b_count,color = 'magenta',linewidth=3)
    limCount = np.max(np.abs(y_pred_count))
    ax[0].plot(y_test, y_pred_count, marker='o', linestyle='',
               color='gray')
    ax[0].set_xlabel(r'$s$',fontsize=25)
    ax[0].set_ylabel(r'$\hat{s}$',fontsize=25)

    # ax[1].set_title('rank',fontsize=25)
    ax[1].set_title(f'rank ($R^2={r2_test_rank:.3}$)',fontsize=25)
    ax[1].plot(x, m_rank*x+b_rank,color = 'orange',linewidth=3)
    limRank = np.max(np.abs(y_pred_rank))
    ax[1].plot(y_test, y_pred_rank, marker='o', linestyle='',
               color='gray')
    ax[1].set_xlabel(r'$s$',fontsize=25)

    mi_count = mutual_info_regression(y_test.reshape(-1,1), y_pred_count)[0]/np.log(2)
    mi_rank = mutual_info_regression(y_test.reshape(-1,1), y_pred_rank)[0]/np.log(2)

    for axi in ax:
        axi.tick_params(axis='x',labelsize=12)
        axi.tick_params(axis='y',labelsize=12)

    lim = np.max([limCount,limRank])
#     fname = 'predictedVtrue_'+ layerName +'_' + tag + f'_{seednum}.png'
#     path = dirName + '/' + fname
    plt.savefig(fname,bbox_inches='tight',dpi=200)
    plt.show()
    plt.close()
    
    fitMetrics = [r2_train_count, r2_test_count, r2_train_rank, r2_test_rank, mi_count, mi_rank]
    data = [y_test, y_pred_count, y_pred_rank]
    
    return fitMetrics, data

# getReconstruction(tvecSub,y_test,y_pred_count,y_pred_rank,layerName,tag,seednum)
def getReconstruction(tvecSub,y_test,y_pred_count,y_pred_rank,layerName,tag,seednum):
    plotDir = 'reconstructions'
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
        
    seedDir = f'{seednum}'
    if not os.path.exists(plotDir + '/' + seedDir):
        os.mkdir(plotDir + '/' + seedDir)    
    
    plotPath = plotDir + '/' + seedDir + '/' + tag
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
        
    fname = plotPath + '/reconstruction_'+ layerName + '.png'
    
    N_test = len(y_test)
    fig, ax = plt.subplots(2,figsize=(9,6))
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
    ax[1].plot(t_test,y_pred_rank,marker='',linestyle='-',color='orange',linewidth=1,alpha=0.8)
    # ax[1].set_ylim([-2,2])
    ax[1].plot(t_test,y_test,marker='',linestyle='-',color='black',label='true',linewidth=3,alpha=0.8)
    ax[1].set_ylabel('stimulus',fontsize=20)
    ax[1].legend(handles=[black_patch, magenta_patch, orange_patch],bbox_to_anchor=(1.25,1.4),fontsize=15)
    ax[1].set_xlim([t_start,t_end])
    ax[1].set_xlabel(r'time (ms)',fontsize=20)
#     fname = 'reconstruct_'+ layerName +'_' + tag + f'_{seednum}.png'
#     path = dirName + '/' + fname
    plt.savefig(fname,bbox_inches='tight',dpi=200)
    plt.show()
    plt.close()
# getStimDist(y_test,y_pred_count,y_pred_rank,layerName,tag,seednum)
def getStimDist(y_test,y_pred_count,y_pred_rank,layerName,tag,seednum):
    plotDir = 'stimDists'
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
        
    seedDir = f'{seednum}'
    if not os.path.exists(plotDir + '/' + seedDir):
        os.mkdir(plotDir + '/' + seedDir)
    
    plotPath = plotDir + '/' + seedDir + '/' + tag
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
        
    fname = plotPath + '/stimDist_'+ layerName + '.png'
    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    for axi in ax:
        axi.tick_params(axis='x',labelsize=12)
        axi.tick_params(axis='y',labelsize=12)
    gray_patch = mpatches.Patch(color='gray', label='true')
    magenta_patch = mpatches.Patch(color='magenta', label='count')
    orange_patch = mpatches.Patch(color='orange', label='rank')
    x_true = ax[0].hist(y_test,density=True,bins=25,alpha=0.7,color='gray')
    ax[1].hist(y_test,density=True,bins=25,alpha=0.7,color='gray',label='true')
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
    ax[0].set_title(f'entropy = {count_ent:.4} bits',fontsize=25,color='magenta')
    ax[1].set_title(f'entropy = {rank_ent:.4} bits',fontsize=25,color='orange')
    ax[0].set_ylabel('probability density',fontsize=20)
    ax[1].set_xlabel('stimulus',fontsize=20)
    ax[0].set_xlabel('stimulus',fontsize=20)
    ax[1].legend(handles=[gray_patch, magenta_patch, orange_patch],bbox_to_anchor=(1.5,0.65),fontsize=15)
#     fname = 'stimDist_'+ layerName +'_' + tag + '.png'
#     path = dirName + '/' + fname
    plt.savefig(fname,bbox_inches='tight',dpi=200)
    plt.show()
    plt.close()
    
    return [true_ent, count_ent, rank_ent]

def plotFit(mem, label, seednum, tag):
    if not os.path.exists('misc'):
        os.mkdir('misc')
    plt.plot(mem[:, 0, 0].cpu(), label="Output")
    plt.plot(label[:, 0, 0].cpu(), '--', label="Target")
    plt.title(tag,fontsize=18)
    plt.xlabel("time",fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("membrane potential",fontsize=18)
    plt.legend(loc='best',fontsize=15)
    fname = tag + f'{seednum}.png'
    path = 'misc/' + fname
    plt.savefig(path,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
