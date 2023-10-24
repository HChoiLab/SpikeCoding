from model import model
from help_funcs import *

pars={}
NE1 = 10
pars['NE1'] = NE1
# resonator parameters
pars['a'] = 0.1
pars['b'] = 0.26
pars['c'] = -60
pars['d'] = -1
Nvec = [50,60,70,80,90]
numN = len(Nvec)
layerNames = ['in', 'E1', 'B']
numLayers = len(layerNames)
T_R = 50
delta_t = 5
train_frac = 0.5

numseeds = 10
r2_tr_c_allseeds = np.zeros((numseeds,numN))
r2_te_c_allseeds = np.zeros((numseeds,numN))
r2_tr_r_allseeds = np.zeros((numseeds,numN))
r2_te_r_allseeds = np.zeros((numseeds,numN))
mi_c_allseeds = np.zeros((numseeds,numN))
mi_r_allseeds = np.zeros((numseeds,numN))
ent_c_allseeds = np.zeros((numseeds,numN))
ent_r_allseeds = np.zeros((numseeds,numN))

for seednum in range(numseeds):
    print(f'seed {seednum}')
    random.seed(seednum)
    np.random.seed(seednum)
    #----- decoding analysis -----
    r2_tr_c_B  = []
    r2_te_c_B  = []
    r2_tr_r_B  = []
    r2_te_r_B  = []
    mi_c_B  = []
    mi_r_B  = []
    ent_c_B = []
    ent_r_B = []
    li = 2 # index of layer to decode
    delay = li*20
    for Ni in range(numN):
        N = Nvec[Ni]
        Nin = N
        NB = N
        Ns = [Nin, NE1, NB]
        pars['Nin'] = Nin
        pars['NB'] = NB
        print(f'N = {N}')
        #----- run network simulation -----
        tvec, sos, spikeTimes, spikeIndices = model(pars)
        tf = tvec[-1]
        dt = tvec[1] - tvec[0]
        makeRaster(Ns, sos, spikeTimes, spikeIndices, Ni)
        #----- decoding analysis -----
        slideVec = np.arange(0,tf-T_R-delay+dt,dt)
        num_times = len(slideVec)
        stimSub = sos[:num_times]
        tvecSub = tvec[:num_times]
        y = np.array(stimSub)
        y = np.reshape(y,(len(y),))
        stim = y

        layerName = layerNames[li]
#         print(layerName)
        N_nrn = Ns[li]
        st = spikeTimes[li]
        si = spikeIndices[li]
        fitMetrics, data = getCorrPlots(st,si,stim,N_nrn,tf,T_R,dt,delta_t,
                                                Ni,delay,layerName)
        r2_tr_c, r2_te_c, r2_tr_r, r2_te_r, mi_c, mi_r = fitMetrics
        y_test, y_pred_count, y_pred_rank = data
        getReconstruction(tvecSub,y_test,y_pred_count,y_pred_rank,layerName,Ni)
        true_ent, count_ent, rank_ent = getStimDist(y_test,y_pred_count,y_pred_rank,layerName,Ni)

        r2_tr_c_B.append(r2_tr_c)
        r2_te_c_B.append(r2_te_c)
        r2_tr_r_B.append(r2_tr_r)
        r2_te_r_B.append(r2_te_r)
        mi_c_B.append(mi_c)
        mi_r_B.append(mi_r)
        ent_c_B.append(count_ent)
        ent_r_B.append(rank_ent)
        
    r2_tr_c_allseeds[seednum,:] = np.array(r2_tr_c_B)
    r2_te_c_allseeds[seednum,:] = np.array(r2_te_c_B)
    r2_tr_r_allseeds[seednum,:] = np.array(r2_tr_r_B)
    r2_te_r_allseeds[seednum,:] = np.array(r2_te_r_B)
    mi_c_allseeds[seednum,:] = np.array(mi_c_B)
    mi_r_allseeds[seednum,:] = np.array(mi_r_B)
    ent_c_allseeds[seednum,:] = np.array(ent_c_B)
    ent_r_allseeds[seednum,:] = np.array(ent_r_B)
        
    print(f'finished decoding all layers for seed {seednum}\n')
    
countR2s_mean_tr = np.mean(r2_tr_c_allseeds,axis=0)
countR2s_sd_tr = np.std(r2_tr_c_allseeds,axis=0)
rankR2s_mean_tr = np.mean(r2_tr_r_allseeds,axis=0)
rankR2s_sd_tr = np.std(r2_tr_r_allseeds,axis=0)
plt.errorbar(NE1vec,countR2s_mean_tr,countR2s_sd_tr,label='count',marker='o',linewidth=3,markersize=10,color='magenta')
plt.errorbar(NE1vec,rankR2s_mean_tr,rankR2s_sd_tr,label='rank',marker='o',linewidth=3,markersize=10,color='orange')
plt.plot(NE1vec,np.zeros(numE1),color='gray',linestyle='--',linewidth=3)
plt.xticks(NE1vec,fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('training set\n decoding accuracy $R^2$',fontsize=15)
plt.xlabel('$N_{E1}$',fontsize=20)
plt.legend(fontsize=15)
plt.savefig('trainR2_v_N_10seed_B.png',bbox_inches='tight',dpi=200)
# plt.show()
plt.close()

countR2s_mean_te = np.mean(r2_te_c_allseeds,axis=0)
countR2s_sd_te = np.std(r2_te_c_allseeds,axis=0)
rankR2s_mean_te = np.mean(r2_te_r_allseeds,axis=0)
rankR2s_sd_te = np.std(r2_te_r_allseeds,axis=0)
plt.errorbar(NE1vec,countR2s_mean_te,countR2s_sd_te,label='count',marker='o',linewidth=3,markersize=10,color='magenta')
plt.errorbar(NE1vec,rankR2s_mean_te,rankR2s_sd_te,label='rank',marker='o',linewidth=3,markersize=10,color='orange')
plt.plot(NE1vec,np.zeros(numE1),color='gray',linestyle='--',linewidth=3)
plt.xticks(NE1vec,fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('test set\n decoding accuracy $R^2$',fontsize=15)
plt.xlabel('$N_{E1}$',fontsize=20)
plt.legend(fontsize=15)
plt.savefig('testR2_v_N_10seed_B.png',bbox_inches='tight',dpi=200)
# plt.show()
plt.close()

mi_c_mean = np.mean(mi_c_allseeds,axis=0)
mi_c_sd = np.std(mi_c_allseeds,axis=0)
mi_r_mean = np.mean(mi_r_allseeds,axis=0)
mi_r_sd = np.std(mi_r_allseeds,axis=0)
plt.errorbar(NE1vec, mi_c_mean, mi_c_sd, label='count', marker='o', linewidth=3, markersize=10, color='magenta')
plt.errorbar(NE1vec, mi_r_mean, mi_r_sd, label='rank', marker='o', linewidth=3, markersize=10, color='orange')
# # plt.plot(layerVec,true_ent*np.ones(numLayers),color='gray',linestyle='--',linewidth=3)
plt.xticks(NE1vec,fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel(r'mutual information $I(\hat{s},s)$ (bits)',fontsize=15)
plt.xlabel('$N_{E1}$',fontsize=20)
plt.legend(fontsize=15)
plt.savefig('mi_v_N_10seed_B.png',bbox_inches='tight',dpi=200)
# plt.show()
plt.close()

ent_c_mean = np.mean(ent_c_allseeds,axis=0)
ent_c_sd = np.std(ent_c_allseeds,axis=0)
ent_r_mean = np.mean(ent_r_allseeds,axis=0)
ent_r_sd = np.std(ent_r_allseeds,axis=0)
layerLabels = ['in','E1','B','E2','out']
plt.errorbar(NE1vec, ent_c_mean, ent_c_sd, label='count', marker='o', linewidth=3, markersize=10, color='magenta')
plt.errorbar(NE1vec, ent_r_mean, ent_r_sd, label='rank', marker='o', linewidth=3, markersize=10, color='orange')
plt.plot(NE1vec,true_ent*np.ones(numE1),color='gray',linestyle='--',linewidth=3)
plt.xticks(NE1vec,fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('entropy of\nreconstructed stimulus (bits)',fontsize=15)
plt.xlabel('$N_{E1}$',fontsize=20)
plt.legend(fontsize=15)
plt.savefig('ent_v_N_10seed_B.png',bbox_inches='tight',dpi=200)
# plt.show()
plt.close()
    