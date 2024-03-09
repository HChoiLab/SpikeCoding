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
Nh = int(sys.argv[2])


if not os.path.exists(f'rawData/Nh{Nh}'):
    os.mkdir(f'rawData/Nh{Nh}')

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

Nin = 100
Nout = 100
Ns = [Nin, Nh, Nout]
tag = 'post'

pconvec = [0.3]

for pcon in pconvec:
    
	if not os.path.exists(f'rawData/Nh{Nh}/pcon{pcon}'):
		os.mkdir(f'rawData/Nh{Nh}/pcon{pcon}')

	print(f'pcon = {pcon}')
#	torch.manual_seed(seednum)
#	random.seed(seednum)
#	np.random.seed(seednum)
	# modelName = f'trainedModels/model_Nh{Nh}_seed{seednum}.pth'
	dataset = RegressionDataset(timesteps=num_steps,num_samples=num_samples, mode=mode)

	for seednum in range(numseeds):
		print(f'seed {seednum}')

		if not os.path.exists(f'rawData/Nh{Nh}/pcon{pcon}/seed{seednum}'):
			os.mkdir(f'rawData/Nh{Nh}/pcon{pcon}/seed{seednum}')
#		if not os.path.exists(f'rawData/pcon{pcon}/seed{seednum}'):
#			os.mkdir(f'rawData/pcon{pcon}/seed')

		torch.manual_seed(seednum)
		random.seed(seednum)
		np.random.seed(seednum)

		modelName = f'trainedModels/Nh{Nh}/pcon{pcon}/seed{seednum}.pth'
		model = ConDivNet2dStimSparse(timesteps=num_steps, Nin=Nin, Nh=Nh, Nout=Nout, pcon=pcon).to(device)
		model.load_state_dict(torch.load(modelName))


		with torch.no_grad():
			feature = dataset.features
			label = dataset.labels
			feature = feature.to(device)
			s = label.to(device)
			mem, spk1, spk2, spk3, spk4, spkh, spkout = model(s)
		    
		plot2Dfit(s,mem,seednum,Nh,tag)
# 		makeRaster2D_5layer(s, spk1, spk2, spk3, spk4, spkE1, spkB, spkE2, spkout, mem, seednum, tag)

		if seednum == 0:
			stim = np.array(label[:,:,0])
			np.save(f'rawData/stim.npy',stim)

		spk_in_1 = spk1.detach().numpy()
		spk_in_2 = spk2.detach().numpy()
		spk_in_3 = spk3.detach().numpy()
		spk_in_4 = spk4.detach().numpy()
		spk_h_rec = spkh.detach().numpy()
		spk_out_rec = spkout.detach().numpy()

		in_spks = np.hstack((spk_in_1,spk_in_2,spk_in_3,spk_in_4))
		np.save(f'rawData/Nh{Nh}/pcon{pcon}/seed{seednum}/in_spks.npy',in_spks)
		np.save(f'rawData/Nh{Nh}/pcon{pcon}/seed{seednum}/h_spks.npy',spk_h_rec)

# 		spk_B_rec_re = np.reshape(spk_B_rec,(num_steps,NB))
# 		B_spks = np.array(spk_B_rec_re,dtype=bool)
# 		np.save(f'rawData/seed{seednum}/pcon{pcon}/B_spks.npy',B_spks)

# 		spk_E2_rec_re = np.reshape(spk_E2_rec,(num_steps,NE2))
# 		E2_spks = np.array(spk_E2_rec_re,dtype=bool)
# 		np.save(f'rawData/seed{seednum}/pcon{pcon}/E2_spks.npy',E2_spks)

# 		spk_out_rec_re = np.reshape(spk_out_rec,(num_steps,Nout))
# 		out_spks = np.array(spk_out_rec_re,dtype=bool)
		np.save(f'rawData/Nh{Nh}/pcon{pcon}/seed{seednum}/out_spks.npy',spk_out_rec)



