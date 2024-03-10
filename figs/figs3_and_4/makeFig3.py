import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os

import matplotlib.patches as mpatches

if not os.path.exists('plotDir'):
    os.mkdir('plotDir')
    
delta_t_list = [1,2,5,10,20,30,40,50,60,70,80,90,100]
numDeltas = len(delta_t_list)
numseeds = 25
layers = ['in','E1','B','E2','out']
layerNames = layers

pconvec = [0.3]

for pcon in pconvec:
    fig,ax = plt.subplots(1,5,figsize=(20,4))
    for li in range(len(layers)):
        layer = layers[li]
        layerName = layerNames[li]

        all_R2s = np.zeros((numseeds,numDeltas))
        all_R2s_4Hz = np.zeros((numseeds,numDeltas))
        all_R2s_20Hz = np.zeros((numseeds,numDeltas))
        for seed in range(numseeds):
            for dti in range(numDeltas):
                deltat = delta_t_list[dti]
                path = f'decodeData/seed{seed}/deltat{deltat}'
                ytest_path = path + '/ytest.npy'
                ytest = np.load(ytest_path)
                ypred_path = path + f'/ypred_{layer}.npy'
                ypred = np.load(ypred_path)
                all_R2s[seed,dti] = r2_score(ytest,ypred)

                ################### 4Hz ##########################
                ytest_path = path + '/y4test.npy'
                ytest = np.load(ytest_path)
                ypred_path = path + f'/y4pred_{layer}.npy'
                ypred = np.load(ypred_path)
                all_R2s_4Hz[seed,dti] = r2_score(ytest,ypred)

                ################### 20Hz ##########################
                ytest_path = path + '/y20test.npy'
                ytest = np.load(ytest_path)
                ypred_path = path + f'/y20pred_{layer}.npy'
                ypred = np.load(ypred_path)
                all_R2s_20Hz[seed,dti] = r2_score(ytest,ypred)

        mean_R2s = all_R2s.mean(axis=0)
        sd_R2s = all_R2s.std(axis=0)
        sem_R2s = sd_R2s/np.sqrt(numseeds)
        mean4_R2s = all_R2s_4Hz.mean(axis=0)
        sd4_R2s = all_R2s_4Hz.std(axis=0)
        sem4_R2s = sd4_R2s/np.sqrt(numseeds)
        mean20_R2s = all_R2s_20Hz.mean(axis=0)
        sd20_R2s = all_R2s_20Hz.std(axis=0)  
        sem20_R2s = sd20_R2s/np.sqrt(numseeds)

        ax[li].set_xlim([-3,103])

        ax[li].errorbar(delta_t_list,mean_R2s,sem_R2s,color='black',marker='x',linewidth=3,
                            markersize=8)
        ax[li].errorbar(delta_t_list,mean4_R2s,sem4_R2s,marker='x',color='orange',linewidth=3,
                            markersize=8)
        ax[li].errorbar(delta_t_list,mean20_R2s,sem20_R2s,marker='x',color='green',linewidth=3,
                            markersize=8)
        ax[li].set_ylabel(f'{layerName} layer $R^2$',fontsize=20)
        ax[li].set_xlabel('bin size $\Delta t$ (ms)',fontsize=20)
        ax[li].tick_params(axis='x',labelsize=15)
        ax[li].tick_params(axis='y',labelsize=15)
        if li != 4:
            ax[li].set_ylim([0,1])
    ax[4].plot(delta_t_list,np.zeros(len(delta_t_list)),color='gray',linestyle='--',linewidth=2)
    black = mpatches.Patch(color='black', label='composite')
    orange = mpatches.Patch(color='orange', label='4 Hz')
    green = mpatches.Patch(color='green', label='20 Hz')
    plt.tight_layout()
    ax[2].legend(ncol=3,handles=[black,orange,green],bbox_to_anchor=(1.55,1.3),fontsize=20)
#     fig.suptitle(f'$p = {1-pcon:.2}$', fontsize=20, y=0.94)
    plt.savefig(f'plotDir/R2_v_deltat_freq_5layer.eps',bbox_inches='tight',dpi=200)
    plt.savefig(f'plotDir/R2_v_deltat_freq_5layer.jpg',bbox_inches='tight',dpi=200)
#     plt.tight_layout()
    plt.show()
    plt.close()
    