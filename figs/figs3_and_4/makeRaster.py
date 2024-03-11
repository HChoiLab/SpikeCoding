import numpy as np
import matplotlib.pyplot as plt
from help_funcs import *

if not os.path.exists('plotDir'):
    os.mkdir('plotDir')

stim = np.load(f'rawData/stim.npy')
tf = stim.shape[0]

seednum = 1

in_spks = np.load(f'rawData/seed{seednum}/in_spks.npy')
Nin = in_spks.shape[1]
E1_spks = np.load(f'rawData/seed{seednum}/E1_spks.npy')
NE1 = E1_spks.shape[1]
B_spks = np.load(f'rawData/seed{seednum}/B_spks.npy')
NB = B_spks.shape[1]
E2_spks = np.load(f'rawData/seed{seednum}/E2_spks.npy')
NE2 = E2_spks.shape[1]
out_spks = np.load(f'rawData/seed{seednum}/out_spks.npy')
Nout = out_spks.shape[1]
mem = np.load(f'rawData/seed{seednum}/mem.npy')

stin , siin = np.where(in_spks == True)
stE1 , siE1 = np.where(E1_spks == True)
stB  , siB  = np.where(B_spks == True)
stE2 , siE2 = np.where(E2_spks == True)
stout, siout= np.where(out_spks == True)


fig, ax = plt.subplots(7,figsize=(9,12),gridspec_kw={'height_ratios': [1, 3, 8, 2, 4, 2, 1]})
for i in range(len(ax)):
    ax[i].set_xlim([0,tf])
    ax[i].tick_params(axis='x',labelsize=15)
    ax[i].tick_params(axis='y',labelsize=12)
    if i != 6:
        ax[i].set_xticks([])

ax[0].plot(stim[:,0],color='red')
ax[0].plot(stim[:,1],color='lime')
ax[0].set_ylim([-1,1])
ax[0].set_ylabel('stim',rotation=0, fontsize=18)
ax[0].yaxis.set_label_coords(-0.1,0.2)
ax[1].set_yticks([1,Nin])
ax[1].plot(stin, siin, color='black', marker='o', linestyle='',markersize=1)
ax[1].set_ylim([-1,Nin+1])
ax[1].set_ylabel('input',rotation=0, fontsize=18)
ax[1].yaxis.set_label_coords(-0.11,0.4)
ax[2].set_yticks([1,NE1])
ax[2].plot(stE1, siE1, color='black', marker='o', linestyle='',markersize=0.2)
ax[2].set_ylim([-1,NE1+1])
ax[2].set_ylabel('E1',rotation=0, fontsize=18)
ax[2].yaxis.set_label_coords(-0.11,0.45)
ax[3].set_yticks([0,NB],['1','10'])
ax[3].plot(stB, siB, color='black', marker='o', linestyle='',markersize=1)
ax[3].set_ylim([0,NB])
ax[3].set_ylabel('B',rotation=0, fontsize=18)
ax[3].yaxis.set_label_coords(-0.11,0.4)
ax[4].plot(stE2, siE2, color='black', marker='o', linestyle='',markersize=0.5)
ax[4].set_yticks([1,NE2])
ax[4].set_ylim([-1,NE2+1])
ax[4].set_ylabel('E2',rotation=0, fontsize=18)
ax[4].yaxis.set_label_coords(-0.11,0.45)
ax[5].plot(stout, siout, color='black', marker='o', linestyle='',markersize=1)
ax[5].set_ylim([-1,Nout+1])
ax[5].set_yticks([0,Nout],['1','10'])
ax[5].set_ylabel('output',rotation=0, fontsize=18)
ax[5].yaxis.set_label_coords(-0.11,0.4)
ax[6].plot(mem[:,0],color='red')
ax[6].plot(mem[:,1],color='lime')
ax[6].set_ylim([-1,1])
ax[6].set_xlabel('time (ms)',fontsize=20)
ax[6].set_ylabel('readout',rotation=0, fontsize=18)
ax[6].yaxis.set_label_coords(-0.11,0.3)

plt.savefig('plotDir/5layer_raster.eps',bbox_inches='tight',dpi=200)
plt.savefig('plotDir/5layer_raster.jpg',bbox_inches='tight',dpi=200)
plt.show()
