from model import model
from help_funcs import *

Nin = 50
NE1 = 2000
NB = 50
NE2 = 200
Nout = 50
pars = {}
pars['Nin'] = Nin
pars['NE1'] = NE1
pars['NB'] = NB
pars['NE2'] = NE2
pars['Nout'] = Nout
# resonator parameters
pars['a'] = 0.1
pars['b'] = 0.26
pars['c'] = -60
pars['d'] = -1

Ns = [Nin, NE1, NB, NE2, Nout]
layerNames = ['in', 'E1', 'B', 'E2', 'out']
N_nrnVec = [Nin, NE1, NB, NE2, Nout]
numLayers = len(layerNames)
T_R = 50
delta_t = 5
train_frac = 0.5

numseeds = 10
r2_tr_c_allseeds_pre = np.zeros((numseeds,numLayers))
r2_te_c_allseeds_pre = np.zeros((numseeds,numLayers))
r2_tr_r_allseeds_pre = np.zeros((numseeds,numLayers))
r2_te_r_allseeds_pre = np.zeros((numseeds,numLayers))
mi_c_allseeds_pre = np.zeros((numseeds,numLayers))
mi_r_allseeds_pre = np.zeros((numseeds,numLayers))
ent_c_allseeds_pre = np.zeros((numseeds,numLayers))
ent_r_allseeds_pre = np.zeros((numseeds,numLayers))


for seednum in range(numseeds):
    print(f'seed {seednum}')
    random.seed(seednum)
    np.random.seed(seednum)
    #----- run network simulation -----
    tvec, sos, spikeTimes, spikeIndices = model(pars)
    nt = len(tvec)
    makeRaster(Ns, sos, spikeTimes, spikeIndices, seednum)
    print('\tmade raster')
    tf = tvec[-1]
    dt = tvec[1] - tvec[0]
    #----- decoding analysis -----
    r2_tr_c_all = []
    r2_te_c_all = []
    r2_tr_r_all = []
    r2_te_r_all = []
    mi_c_all = []
    mi_r_all = []
    ent_c_all = []
    ent_r_all = []
    delay = 0
    for li in range(numLayers):
        slideVec = np.arange(0,tf-T_R-delay+dt,dt)
        num_times = len(slideVec)
        stimSub = sos[:num_times]
        tvecSub = tvec[:num_times]
        y = np.array(stimSub)
        
        if seednum == 0:
            if not os.path.exists('dataDir'):
                os.mkdir('dataDir')
            filename = f'dataDir/y_{delay}.txt'
            file = open(filename,'w+')
            np.savetxt(file,y)
            file.close()

        y = np.reshape(y,(len(y),))
        stim = y
        
        layerName = layerNames[li]
        print(layerName)
        N_nrn = Ns[li]
        st = spikeTimes[li]
        si = spikeIndices[li]
        if li == 1: # decode subset of neurons for E1 layer
            N_nrn = 200
        fitMetrics, data = getCorrPlots(st,si,stim,N_nrn,tf,T_R,dt,delta_t,
                                                seednum,delay,layerName)
        r2_tr_c, r2_te_c, r2_tr_r, r2_te_r, mi_c, mi_r = fitMetrics
        y_test, y_pred_count, y_pred_rank = data
        getReconstruction(tvecSub,y_test,y_pred_count,y_pred_rank,layerName,seednum)
        true_ent, count_ent, rank_ent = getStimDist(y_test,y_pred_count,y_pred_rank,layerName,seednum)

        r2_tr_c_all.append(r2_tr_c)
        r2_te_c_all.append(r2_te_c)
        r2_tr_r_all.append(r2_tr_r)
        r2_te_r_all.append(r2_te_r)
        mi_c_all.append(mi_c)
        mi_r_all.append(mi_r)
        ent_c_all.append(count_ent)
        ent_r_all.append(rank_ent)
        delay += 20
        
    r2_tr_c_allseeds_pre[seednum,:] = np.array(r2_tr_c_all)
    r2_te_c_allseeds_pre[seednum,:] = np.array(r2_te_c_all)
    r2_tr_r_allseeds_pre[seednum,:] = np.array(r2_tr_r_all)
    r2_te_r_allseeds_pre[seednum,:] = np.array(r2_te_r_all)
    mi_c_allseeds_pre[seednum,:] = np.array(mi_c_all)
    mi_r_allseeds_pre[seednum,:] = np.array(mi_r_all)
    ent_c_allseeds_pre[seednum,:] = np.array(ent_c_all)
    ent_r_allseeds_pre[seednum,:] = np.array(ent_r_all)
        
    print(f'finished decoding all layers for seed {seednum}\n')

    
layerVec = np.arange(0,numLayers,1)
layerLabels = layerNames
countR2s_mean_tr = np.mean(r2_tr_c_allseeds_pre,axis=0)
countR2s_sd_tr = np.std(r2_tr_c_allseeds_pre,axis=0)
rankR2s_mean_tr = np.mean(r2_tr_r_allseeds_pre,axis=0)
rankR2s_sd_tr = np.std(r2_tr_r_allseeds_pre,axis=0)
plt.errorbar(layerVec,countR2s_mean_tr,countR2s_sd_tr,label='count',marker='o',linewidth=3,markersize=10,color='magenta')
plt.errorbar(layerVec,rankR2s_mean_tr,rankR2s_sd_tr,label='rank',marker='o',linewidth=3,markersize=10,color='orange')
plt.plot(layerVec,np.zeros(numLayers),color='gray',linestyle='--',linewidth=3)
plt.xticks(layerVec,layerLabels,fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('training set\n decoding accuracy $R^2$',fontsize=15)
plt.xlabel('layer',fontsize=15)
plt.legend(fontsize=15)
plt.savefig('trainR2_v_layer_10seeds.png',bbox_inches='tight',dpi=200)
# plt.show()
plt.close()

countR2s_mean_te = np.mean(r2_te_c_allseeds_pre,axis=0)
countR2s_sd_te = np.std(r2_te_c_allseeds_pre,axis=0)
rankR2s_mean_te = np.mean(r2_te_r_allseeds_pre,axis=0)
rankR2s_sd_te = np.std(r2_te_r_allseeds_pre,axis=0)
plt.errorbar(layerVec,countR2s_mean_te,countR2s_sd_te,label='count',marker='o',linewidth=3,markersize=10,color='magenta')
plt.errorbar(layerVec,rankR2s_mean_te,rankR2s_sd_te,label='rank',marker='o',linewidth=3,markersize=10,color='orange')
plt.plot(layerVec,np.zeros(numLayers),color='gray',linestyle='--',linewidth=3)
plt.xticks(layerVec,layerLabels,fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('test set\n decoding accuracy $R^2$',fontsize=15)
plt.xlabel('layer',fontsize=15)
plt.legend(fontsize=15)
plt.savefig('testR2_v_layer_10seeds.png',bbox_inches='tight',dpi=200)
# plt.show()
plt.close()

mi_c_mean = np.mean(mi_c_allseeds_pre,axis=0)
mi_c_sd = np.std(mi_c_allseeds_pre,axis=0)
mi_r_mean = np.mean(mi_r_allseeds_pre,axis=0)
mi_r_sd = np.std(mi_r_allseeds_pre,axis=0)
plt.errorbar(layerVec, mi_c_mean, mi_c_sd, label='count', marker='o', linewidth=3, markersize=10, color='magenta')
plt.errorbar(layerVec, mi_r_mean, mi_r_sd, label='rank', marker='o', linewidth=3, markersize=10, color='orange')
# # plt.plot(layerVec,true_ent*np.ones(numLayers),color='gray',linestyle='--',linewidth=3)
plt.xticks(layerVec,layerLabels,fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel(r'mutual information $I(\hat{s},s)$ (bits)',fontsize=15)
plt.xlabel('layer',fontsize=15)
plt.legend(fontsize=15)
plt.savefig('mi_v_layer_10seeds.png',bbox_inches='tight',dpi=200)
# plt.show()
plt.close()

ent_c_mean = np.mean(ent_c_allseeds_pre,axis=0)
ent_c_sd = np.std(ent_c_allseeds_pre,axis=0)
ent_r_mean = np.mean(ent_r_allseeds_pre,axis=0)
ent_r_sd = np.std(ent_r_allseeds_pre,axis=0)
layerVec = np.arange(0,numLayers,1)
layerLabels = ['in','E1','B','E2','out']
plt.errorbar(layerVec, ent_c_mean, ent_c_sd, label='count', marker='o', linewidth=3, markersize=10, color='magenta')
plt.errorbar(layerVec, ent_r_mean, ent_r_sd, label='rank', marker='o', linewidth=3, markersize=10, color='orange')
plt.plot(layerVec,true_ent*np.ones(numLayers),color='gray',linestyle='--',linewidth=3)
plt.xticks(layerVec,layerLabels,fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('entropy of\nreconstructed stimulus (bits)',fontsize=15)
plt.xlabel('layer',fontsize=15)
plt.legend(fontsize=15)
plt.savefig('ent_v_layer_10seeds.png',bbox_inches='tight',dpi=200)
# plt.show()
plt.close()