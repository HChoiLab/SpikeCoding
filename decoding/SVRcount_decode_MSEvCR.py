# this function generates neural data from a convergent network of spiking neurons
# receiving a sinusoidal input with added noise. It returns the response of the
# network as a sequence of 0's and 1's (neural_data) along with the corresponding 
# stimulus (y).

# Written by Zach Mobille on February 10, 2023

######################### IMPORTS ##############################
from brian2 import *
#import matplotlib.gridspec as gridspec
from Neural_Decoding.preprocessing_funcs import bin_spikes
prefs.codegen.target = 'numpy'
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
# from SVR_temporal_decoding import *
#################################################################

binSize = 40 # delta tau
bins_before = 1
# CR = 10
# fname1 = f'CR{CR}_temporal1_c5.png'
# fname2 = f'CR{CR}_temporal2_c5.png'

########### FUNCTION ###########################################
def generateData(Vt01vec, Vt02vec, Vthvec, inputNum):
    """
    This function simulates a convergent network of spiking LIF neurons
    and returns the response as the spike rate in time of the output neuron
    and the corresponding stimulus

    Parameters:
    'inputNum' (positive integer) = number of spiking neurons in input layer
    'Vt01vec' is vector of voltage threshold equilibria for input group 1
    'Vt02vec' is vector of voltage threshold equilibria for input group 2
    """
    start_scope()
    timeStep = defaultclock.dt
    numH = len(Vthvec)
    # 2D LIF parameters
    duration = 4000*ms
    tau = 10*ms
    taut= 5*ms
    delta_Vt = 5*mV
    delta_V  = 10*mV
    numOut = 10
    Vr = -70*mV
    Vt0= -50*mV
    tau_dum = 1*ms
    # vectors of voltage threshold equilibria for each population
    Vt01vec = Vt01vec*mV
    Vt02vec = Vt02vec*mV
    Vthvec = Vthvec*mV
    #shift = np.random.uniform(0, 1000)*ms # random shift of complex input
    
    # stimulus parameters
    f = 4*Hz # frequency of sinusoidal input
    a = 1*mV # amplitude of s(t)
    A = (2*pi*f*tau_dum)*a # amplitude of ds/dt
    
    # CPG input parameters
    CPGfreq = 20*Hz
    CPGamp = 2.*mV

    ##### 2D LIF #####
    # input equations
    # neuron group 1 (with mean Vt0 mu1)
    eqsI1 = '''
    ds/dt = int(t > duration/2)*A*cos(2*pi*f*t)/tau_dum + int(t <= duration/2)*A*cos(2*pi*f*t)/tau_dum + A/tau_dum**.5*xi : volt
    dV/dt  = (-V + s + Vt01 + CPGamp*sin(2*pi*CPGfreq*t))/tau : volt
    dVt/dt = -(Vt - Vt01)/taut : volt
    dVt01/dt = 0*mV/tau_dum : volt
    '''

    # neuron group 2 (with mean Vt0 mu2)
    eqsI2 = '''
    ds/dt = int(t > duration/2)*A*cos(2*pi*f*t)/tau_dum + int(t <= duration/2)*A*cos(2*pi*f*t)/tau_dum + A/tau_dum**.5*xi : volt
    dV/dt  = (-V - s + Vt02 + CPGamp*sin(2*pi*CPGfreq*t))/tau : volt
    dVt/dt = -(Vt-Vt02)/taut : volt
    dVt02/dt = 0*mV/tau_dum : volt
    '''
    
    # neuron group H ('hidden' units)
    eqsH = '''
    dV/dt  = -(V-Vt0h)/tau : volt
    dVt/dt = -(Vt-Vt0h)/taut : volt
    dVt0h/dt = 0*mV/tau_dum : volt
    '''

    # output equations
    eqsO = '''
    dV/dt  = -(V-Vt0)/tau : volt
    dVt/dt = -(Vt-Vt0)/taut : volt
    '''
    

    resetLIF = '''
    V = Vr
    Vt += delta_Vt
    '''
    
    
    # input neurons
    # group 1
    I1 = NeuronGroup(int(inputNum/2), eqsI1, threshold='V>Vt', refractory=2*ms, reset=resetLIF, method='euler') # 'I' for "input"
    I1.s = 0*mV # stimulus starts at 0
    I1.V = 'rand()*(Vt01-Vr)+Vr' # random initial states
    I1.Vt = Vt0
    I1.Vt01 = Vt01vec

    # group 2
    I2 = NeuronGroup(int(inputNum/2), eqsI2, threshold='V>Vt', refractory=2*ms, reset=resetLIF, method='euler') # 'I' for "input"
    I2.s = 0*mV # stimulus starts at 0
    I2.V = 'rand()*(Vt02-Vr)+Vr' # random initial states
    I2.Vt = Vt0
    I2.Vt02 = Vt02vec
    
    # hidden neurons
    H = NeuronGroup(numH, eqsH, threshold='V>Vt', refractory=2*ms, reset=resetLIF, method='euler') # 'I' for "input"
    H.V = 'rand()*(Vt0h-Vr)+Vr' # random initial states
    H.Vt = Vt0
    H.Vt0h = Vthvec
    
    # output neuron
    O = NeuronGroup(numOut, eqsO, threshold='V>Vt', refractory=2*ms, reset=resetLIF, method='euler')
    O.V = 'rand()*(Vt0-Vr)+Vr' # random initial state
    O.Vt = Vt0

    # Monitor the activity
    smon = StateMonitor(I1, 's', True)
    #I1mon = SpikeMonitor(I1, 'V')
    #I1rateMon = PopulationRateMonitor(I1) # monitor I1's spike rate
    #I2mon = SpikeMonitor(I2, 'V')
    #I2rateMon = PopulationRateMonitor(I2) # monitor I2's spike rate
    Omon = SpikeMonitor(O, 'V')
    #OrateMon = PopulationRateMonitor(O) # monitor the output's spike rate

    
    # specify synapses from input layer 1 to output neuron
    I1_H = Synapses(I1, H, on_pre='V_post += delta_V')
    I1_H.connect(p=0.4)
#     I1_H.connect(i=list(np.arange(int(inputNum/2))), j=list(np.arange(numH)))

    # specify synapses from input layer 1 to output neuron
    I2_H = Synapses(I2, H, on_pre='V_post -= delta_V')
    I2_H.connect(p=0.4)
#     I2_H.connect(i=list(np.arange(int(inputNum/2))), j=list(np.arange(numH)))
    
    # specify synapses from hidden layer to output neuron
    H_O = Synapses(H, O, on_pre='V_post += delta_V')
    H_O.connect()

    # run the simulation for a duration 'duration'
    run(duration)
    
    #fig, axes = plt.subplots(6,1,figsize = (9,12),gridspec_kw = {'height_ratios':[1,1,2,2,1,1]})
    
    dt = timeStep/ms
    tmin = 0.
    tmax = duration/ms
    tstep= binSize # spike bin size
    # neural_data has dimension (total number of time bins) X (number of neurons)
    spikeTimes = np.array([np.sort(Omon.t/ms)])
#     print(np.shape(Omon.t/ms))
#     print(Omon.t/ms)
#     print(Omon.i)
#     plt.plot(Omon.t/ms,Omon.i,'.')
#     plt.xlabel('time (ms)',fontsize=20)
#     plt.ylabel('neuron ID',fontsize=20)
#     plt.savefig('rast.png')
    for nrn in range(1):
        st = np.array([Omon.t[Omon.i==nrn]/ms])
        neural_data = bin_spikes(st,tstep,tmin,tmax+tstep)[:,0]
        #print(binTrain)
#         print(st)
#     neural_data = bin_spikes(spikeTimes,tstep,tmin,tmax+tstep)[:,0]
#     print(neural_data)
    #print('np.shape(neural_data) = '+str(np.shape(neural_data)))
    # output data y is I(t) AKA s
    y = []
    t = tmin
    idt = tstep
    while t < tmax:
        app = smon.s[0][int(t/dt)]/mV
        y.append(app)
        t += tstep
    return neural_data, y
    
################ END FUNCTION ##########################################

hNumVec = [50,75,100,125,150]
MSEvec = np.zeros(len(hNumVec))
k = 0
for hNum in hNumVec:
    print('k = '+str(k))
    # number of neurons in input layer
    inputNum = 10
    hNum = 100
    # these are all in millivolts
    mu1 = -60
    mu2 = -40
    sig = 6
    # generate the random voltage threshold equilibria
    Vt01vec = np.random.rand(inputNum//2)*sig + mu1
    Vt02vec = np.random.rand(inputNum//2)*sig + mu2
    Vthvec  = np.random.uniform(low=-70, high=-40, size=hNum)
    result = generateData(Vt01vec, Vt02vec, Vthvec, inputNum)
    # plt.plot(result[1])
    # plt.savefig('plot.png')
    # print(np.shape(result[0]))
    # print(np.shape(result[0]))
    # print(len(result[0])==len(result[1]))


    #Import standard packages
    import numpy as np
    import matplotlib.pyplot as plt
    #%matplotlib inline
    from scipy import io
    from scipy import stats
    from sklearn.metrics import mean_squared_error

    #Import function to get the covariate matrix that includes spike history from previous bins
    from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

    #Import metrics
    from Neural_Decoding.metrics import get_R2
    from Neural_Decoding.metrics import get_rho

    #Import decoder function
    from Neural_Decoding.decoders import SVRDecoder

    #Import the function that generates the data
    # from generate_temporal_data import generateData
    # from generate_temporal_data import binSize

    ####### MAIN PART OF CODE #######
    ########## GLOBAL PARAMETERS ###########
    # # number of neurons in input layer
    # inputNum = CR
    # # these are all in millivolts
    # mu1 = -60
    # mu2 = -40
    # sig = 6
    # # generate the random voltage threshold equilibria
    # Vt01vec = np.random.rand(inputNum//2)*sig + mu1
    # Vt02vec = np.random.rand(inputNum//2)*sig + mu2

    N = 10 # number of samples (training + testing)
    time = 100 # length of each trial in time bins

    # bins_before=20 #How many bins of neural data prior to the output are used for decoding (10)
    bins_current=0 #Whether to use concurrent time bin of neural data
    bins_after=0 #How many bins of neural data after the output are used for decoding (10)

    #neural_data = np.empty(1000, dtype=int)
    neural_data = np.zeros((time, N))
    y = np.zeros((time, N))

    # generate the data
    for n in range(N):
        # generate the data
        x, r = generateData(Vt01vec=Vt01vec, Vt02vec=Vt02vec, Vthvec=Vthvec, inputNum=inputNum)
        neural_data[:,n] = x
        y[:,n] = r
    #print('shape of neural_data = '+str(np.shape(neural_data)))
    #print('shape of y = '+str(np.shape(y)))

    # Format data
    # Function to get the covariate matrix that includes spike history from previous bins
    X = get_spikes_with_history(neural_data,bins_before,bins_after,bins_current)

    #print('shape of X = '+str(np.shape(X)))

    # Format for Wiener Filter, Wiener Cascade, XGBoost, and Dense Neural Network
    #Put in "flat" format, so each "neuron / time" is a single feature
    X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))

    #print('shape of X_flat = '+str(np.shape(X_flat)))


    #Set what part of data should be part of the training/testing/validation sets
    training_range=[0, 0.5]
    testing_range=[0.5, 0.83]
    valid_range=[0.83, 1]

    num_examples=X.shape[0]

    #Note that each range has a buffer of bins_before" bins at the beginning, and "bins_after" bins at the end
    #This makes it so that the different sets don't include overlapping neural data
    training_set=np.arange(int(np.round(training_range[0]*num_examples))+bins_before,int(np.round(training_range[1]*num_examples))-bins_after)
    testing_set=np.arange(int(np.round(testing_range[0]*num_examples))+bins_before,int(np.round(testing_range[1]*num_examples))-bins_after)
    valid_set=np.arange(int(np.round(valid_range[0]*num_examples))+bins_before,int(np.round(valid_range[1]*num_examples))-bins_after)


    #Get training data
    X_train=X[training_set,:,:]
    X_flat_train=X_flat[training_set,:]
    y_train=y[training_set,:]

    #Get testing data
    X_test=X[testing_set,:,:]
    X_flat_test=X_flat[testing_set,:]
    y_test=y[testing_set,:]

    #Get validation data
    X_valid=X[valid_set,:,:]
    X_flat_valid=X_flat[valid_set,:]
    y_valid=y[valid_set,:]

    #Z-score "X" inputs. 
    X_train_mean=np.nanmean(X_train,axis=0)
    X_train_std=np.nanstd(X_train,axis=0)
    X_train=(X_train-X_train_mean)/X_train_std
    X_test=(X_test-X_train_mean)/X_train_std
    X_valid=(X_valid-X_train_mean)/X_train_std

    #Z-score "X_flat" inputs. 
    X_flat_train_mean=np.nanmean(X_flat_train,axis=0)
    X_flat_train_std=np.nanstd(X_flat_train,axis=0)
    X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
    X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std
    X_flat_valid=(X_flat_valid-X_flat_train_mean)/X_flat_train_std

    #Zero-center outputs
    y_train_mean=np.mean(y_train,axis=0)
    y_train=y_train-y_train_mean
    y_test=y_test-y_train_mean
    y_valid=y_valid-y_train_mean

    #The SVR works much better when the y values are normalized, so we first z-score the y values
    #They have previously been zero-centered, so we will just divide by the stdev (of the training set)
    y_train_std=np.nanstd(y_train,axis=0)
    y_zscore_train=y_train/y_train_std
    #y_zscore_test=y_test/y_train_std
    #y_zscore_valid=y_valid/y_train_std


    c = 4. # regularization parameter, float
    #Declare model
    model_svr=SVRDecoder(C=c) # originally C=5

    #Fit model
    model_svr.fit(X_flat_train, y_zscore_train)

    #Get predictions
    y_zscore_test_predicted_svr = model_svr.predict(X_flat_test)

    #Get metric of fit
    #R2s_svr=get_R2(y_zscore_valid,y_zscore_valid_predicted_svr)
    #print('R2s:', R2s_svr)

    #print('\nlength of y_zscore_test_predicted_svr[:,0] = '+str(len(y_zscore_test_predicted_svr[:,0])))
    #print('length of y_valid[:,0] = '+str(len(y_valid[:,0])))

#     print('shape of y_test = '+str(np.shape(y_test)))

    mseVec = []
    for i in range(np.shape(y_test)[1]):
        true = y_test[:,i]+y_train_mean[i]
        predicted = y_zscore_test_predicted_svr[:,0]*y_train_std[0]
        mse = mean_squared_error(true, predicted) # metric of fit
        mseVec.append(mse)
#     print('average mse = '+str(np.mean(mseVec)))
    MSEvec[k] = np.mean(mseVec)
    k += 1
print('MSEvec = '+str(MSEvec))
# # do the plot for 0th test set
# #tVec = np.linspace(0,,len(y_test[:,0]))
# plt.xlabel('time bin',fontsize = 15)
# plt.ylabel('stimulus', fontsize = 15)
# true = y_test[:,0]+y_train_mean[0]
# predicted = y_zscore_test_predicted_svr[:,0]*y_train_std[0]
# mse = mean_squared_error(true, predicted) # metric of fit
# tVec = np.linspace(0,int((len(true)+1)/10),len(true))
# plt.plot(true,color='black',label='true',linewidth=5)
# plt.plot(predicted, color='magenta', label='predicted', linewidth=5)
# # plt.plot(tVec,true,color='black',label='true',linewidth=5)
# # plt.plot(tVec,predicted, color='magenta', label='predicted', linewidth=5)
# plt.title('c = '+str(c)+'\nmse = '+str(mse))
# plt.legend(fontsize=12)
# plt.savefig('plot1_bins_before20.png',bbox_inches='tight')
# plt.show()
# plt.close()

# # do it for 1st test set
# true = y_test[:,1]+y_train_mean[1]
# predicted = y_zscore_test_predicted_svr[:,1]*y_train_std[1]
# mse = mean_squared_error(true, predicted) # metric of fit
# plt.xlabel('time bin',fontsize = 15)
# plt.ylabel('stimulus', fontsize = 15)
# plt.plot(true,color='black',label='true',linewidth=5)
# plt.plot(predicted, color='magenta', label='predicted', linewidth=5)
# # plt.plot(tVec,true,color='black',label='true',linewidth=5)
# # plt.plot(tVec,predicted, color='magenta', label='predicted', linewidth=5)
# plt.title('c = '+str(c)+'\nmse = '+str(mse))
# plt.legend(fontsize=12)
# plt.savefig('plot2_bins_before20.png',bbox_inches='tight')
# plt.show()
