import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import matplotlib.patches as mpatches

if not os.path.exists('plotDir'):
    os.mkdir('plotDir')
    
delta_t_list = [1,2,5,10,20,30,40,50,60,70,80,90,100]
numDeltas = len(delta_t_list)
numseeds = 10
Nhvec = [10,1000]
layers = ['in','h','out']
layerNames = ['input','hidden','output']

pconvec = [0.3]

for pcon in pconvec:
    fig,ax = plt.subplots(3,2,figsize=(8,12))
    ax[0,0].set_title('bottleneck ($N_h=10$)',fontsize=15)
    ax[0,1].set_title('expansion ($N_h=1000$)',fontsize=15)
    for li in range(len(layers)):
        layer = layers[li]
        layerName = layerNames[li]
        for nhi in range(len(Nhvec)):
            nh = Nhvec[nhi]
            all_R2s = np.zeros((numseeds,numDeltas))
            for seed in range(numseeds):
                for dti in range(numDeltas):
                    deltat = delta_t_list[dti]
                    path = f'decodeData/Nh{nh}/pcon{pcon}/seed{seed}/deltat{deltat}'
                    ytest_path = path + '/ytest.npy'
                    ytest = np.load(ytest_path)
                    ypred_path = path + f'/ypred_{layer}.npy'
                    ypred = np.load(ypred_path)
                    all_R2s[seed,dti] = r2_score(ytest,ypred)
            mean_R2s = all_R2s.mean(axis=0)
            maxi = np.argmax(mean_R2s)
            high = mean_R2s[maxi]
            low = mean_R2s[-1]
            ax[li,nhi].hlines(high,-3,delta_t_list[maxi],linestyle='--',color='blue',linewidth=2)
            ax[li,nhi].hlines(low,-3,delta_t_list[-1],linestyle='--',color='magenta',linewidth=2)
            ax[li,nhi].vlines(delta_t_list[maxi],high,low,linestyle='--',color='gray',linewidth=2)
            ax[li,nhi].set_xlim([-3,103])
            ax[li,nhi].text(22,0.18,f'$\Delta R^2={high-low:.2}$',fontsize=15,color='gray')
            
            sd_R2s = all_R2s.std(axis=0)
            sem_R2s = sd_R2s/np.sqrt(numseeds)
            ax[li,nhi].errorbar(delta_t_list,mean_R2s,sem_R2s,color='black',marker='x',linewidth=3,markersize=8,zorder=1)
            ax[li,nhi].plot(delta_t_list[maxi],high,color='blue',marker='o',markersize=8,zorder=2)
            ax[li,nhi].plot(delta_t_list[-1],low,color='magenta',marker='o',markersize=8,zorder=2)
            if nhi == 0:
                ax[li,nhi].set_ylabel(f'{layerName} layer $R^2$',fontsize=18)
            if li == 2:
                ax[li,nhi].set_xlabel('bin size $\Delta t$ (ms)',fontsize=18)
            ax[li,nhi].tick_params(axis='x',labelsize=12)
            ax[li,nhi].tick_params(axis='y',labelsize=12)
            ax[li,nhi].set_ylim([0,1])
    blue = mpatches.Patch(color='blue', label='spike timing')
    magenta = mpatches.Patch(color='magenta', label='spike count')
    ax[0,0].legend(ncol=2,handles=[blue,magenta],bbox_to_anchor=(2.05,1.35),fontsize=18)
    
    plt.savefig(f'plotDir/R2_v_deltat_pcon{1-pcon:.2}.eps',bbox_inches='tight',dpi=200)
    plt.savefig(f'plotDir/R2_v_deltat_pcon{1-pcon:.2}.jpg',bbox_inches='tight',dpi=200)
    plt.show()
    plt.close()