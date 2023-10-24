# imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import itertools
import random
import statistics
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import os
from bayes_opt import BayesianOptimization


def getRankLog(st,delta_t,t1,T_R):
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
    if rank > 0:
        rank = np.log2(rank)
    return rank


def getSlidingCountRankLogVecs(spikeTimes,spikeIndices,N_nrn,tf,T_R,dt,delta_t,delay):
    slideVecTemp = np.arange(delay,tf-T_R+dt,dt)
    countVec = np.zeros((N_nrn,len(slideVecTemp)))
    rankVec = np.zeros((N_nrn,len(slideVecTemp)))
    for ni in range(N_nrn):
        st = spikeTimes[spikeIndices==ni]
        ti = 0 # time index
        for t in slideVecTemp:
            t1 = t
            t2 = t+T_R
            sub_st = st[st > t1]
            sub_st = sub_st[sub_st < t2]
            countVec[ni][ti] = len(sub_st)
            rankVec[ni][ti] = getRankLog(st,delta_t,t1,T_R)
            ti+=1
    
    return [countVec, rankVec]


def makeRaster(Ns, stim, spikeTimes, spikeIndices, seednum):
#     spikeTimes = [stin, stE1, stB, stE2, stout]
#     spikeIndices = [siin, siE1, siB, siE2, siout]
    stin, stE1, stB, stE2, stout = spikeTimes
    siin, siE1, siB, siE2, siout = spikeIndices
    Nin, NE1, NB, NE2, Nout = Ns
    
    if not os.path.exists('rasters'):
        os.mkdir('rasters')
    fig, ax = plt.subplots(6,figsize=(9,12),gridspec_kw={'height_ratios': [1, 2, 8, 2, 4, 2]})

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
    ax[1].plot(stin,siin,marker='o',linestyle='',markersize=0.5, color='black')
    ax[1].set_ylabel('in',fontsize=18)
    ax[1].set_ylim([-1,Nin+1])
    ax[2].plot(stE1,siE1,marker='o',linestyle='',markersize=0.5, color='black')
    ax[2].set_ylabel('E1',fontsize=18)
    ax[2].set_ylim([-1,NE1+1])
    ax[3].plot(stB,siB,marker='o',linestyle='',markersize=0.5, color='black')
    ax[3].set_ylabel('B',fontsize=18)
    ax[3].set_ylim([-1,NB+1])
    ax[4].plot(stE2,siE2,marker='o',linestyle='',markersize=0.5, color='black')
    ax[4].set_ylim([-1,NE2+1])
    ax[4].set_ylabel('E2',fontsize=18)
    ax[5].plot(stout,siout,marker='o',linestyle='',markersize=0.5, color='black')
    ax[5].set_ylabel('out',fontsize=18)
    ax[5].set_ylim([-1,Nout+1])
    fname = f'{seednum}.png'
    path = 'rasters/' + fname
    plt.savefig(path,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()

def getCorrPlots(st,si,stim,N_nrn,tf,T_R,dt,delta_t,seednum,delay,layerName):
    print('\tgetting count and rankvecs...')
    countVec, rankVec = getSlidingCountRankLogVecs(st,si,N_nrn,tf,T_R,dt,delta_t,delay)
    
    dataDir = 'dataDir'
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
        
    seedDir = f'{seednum}'
    if not os.path.exists(dataDir + '/' + seedDir):
        os.mkdir(dataDir + '/' + seedDir)
    
    dataPath = dataDir + '/' + seedDir
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
        
    plotPath = plotDir + '/' + seedDir
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
        
    fname = plotPath + '/predictedVtrue_'+ layerName + '.png'
    
    y = stim
    XC = countVec.T
    XR = rankVec.T
    
    print('\tcount optimization ...')
#     print(f'y.shape = {y.shape}')
#     print(f'XC.shape = {XC.shape}')
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
#     print('made it to plotting')
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    x = y_test
    ax[0].set_title(f'count ($R^2={r2_test_count:.3}$)',fontsize=25)
    ax[0].plot(x, m_count*x+b_count,color = 'magenta',linewidth=3)
    limCount = np.max(np.abs(y_pred_count))
    ax[0].plot(y_test, y_pred_count, marker='o', linestyle='',
               color='gray',alpha=0.3)
    ax[0].set_xlabel(r'$s$',fontsize=25)
    ax[0].set_ylabel(r'$\hat{s}$',fontsize=25)

    # ax[1].set_title('rank',fontsize=25)
    ax[1].set_title(f'rank ($R^2={r2_test_rank:.3}$)',fontsize=25)
    ax[1].plot(x, m_rank*x+b_rank,color = 'orange',linewidth=3)
    limRank = np.max(np.abs(y_pred_rank))
    ax[1].plot(y_test, y_pred_rank, marker='o', linestyle='',
               color='gray',alpha=0.3)
    ax[1].set_xlabel(r'$s$',fontsize=25)

    mi_count = mutual_info_regression(y_test.reshape(-1,1), y_pred_count)[0]/np.log(2)
    mi_rank = mutual_info_regression(y_test.reshape(-1,1), y_pred_rank)[0]/np.log(2)
#     print('calculated mutual info')
    for axi in ax:
        axi.tick_params(axis='x',labelsize=12)
        axi.tick_params(axis='y',labelsize=12)

    lim = np.max([limCount,limRank])
#     fname = 'predictedVtrue_'+ layerName +'_' + tag + f'_{seednum}.png'
#     path = dirName + '/' + fname
    plt.savefig(fname,bbox_inches='tight',dpi=200)
#     plt.show()
    plt.close()
    
    fitMetrics = [r2_train_count, r2_test_count, r2_train_rank, r2_test_rank, mi_count, mi_rank]
    data = [y_test, y_pred_count, y_pred_rank]
#     print('finished plotting')
    return fitMetrics, data

# getReconstruction(tvecSub,y_test,y_pred_count,y_pred_rank,layerName,tag,seednum)
def getReconstruction(tvecSub,y_test,y_pred_count,y_pred_rank,layerName,seednum):
    plotDir = 'reconstructions'
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
        
    seedDir = f'{seednum}'
    if not os.path.exists(plotDir + '/' + seedDir):
        os.mkdir(plotDir + '/' + seedDir)    
    
    plotPath = plotDir + '/' + seedDir
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
#     plt.show()
    plt.close()
# getStimDist(y_test,y_pred_count,y_pred_rank,layerName,tag,seednum)
def getStimDist(y_test,y_pred_count,y_pred_rank,layerName,seednum):
    
    numbins = 15
    
    plotDir = 'stimDists'
    if not os.path.exists(plotDir):
        os.mkdir(plotDir)
        
    seedDir = f'{seednum}'
    if not os.path.exists(plotDir + '/' + seedDir):
        os.mkdir(plotDir + '/' + seedDir)
    
    plotPath = plotDir + '/' + seedDir + '/'
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
    x_true = ax[0].hist(y_test,density=True,bins=numbins,alpha=0.7,color='gray')
    ax[1].hist(y_test,density=True,bins=numbins,alpha=0.7,color='gray',label='true')
    # true stimulus
    px_true = x_true[0]
    dx_true = np.diff(x_true[1])
    p_true = px_true*dx_true
    true_ent = np.nansum(-p_true*np.log2(p_true))
#     print(true_ent)
    # count reconstruction
    x_pred_count = ax[0].hist(y_pred_count,density=True,bins=numbins,alpha=0.7,color='magenta')
    px_count = x_pred_count[0]
    dx_count = np.diff(x_pred_count[1])
    p_count = px_count*dx_count
    count_ent = np.nansum(-p_count[p_count>0]*np.log2(p_count[p_count>0]))
    # rank reconstruction
    x_pred_rank = ax[1].hist(y_pred_rank,density=True,bins=numbins,alpha=0.7,color='orange')
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
#     plt.show()
    plt.close()
    
    return [true_ent, count_ent, rank_ent]
