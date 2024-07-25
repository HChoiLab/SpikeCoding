from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import os
import sys

shuffle=True

Nh = int(sys.argv[1])
stimseed = int(sys.argv[2])
modelseed= int(sys.argv[3])
deltat = int(sys.argv[4])

pcon=0.3
Nin=100
Nout=100

try:
    import keras
    keras_v1=int(keras.__version__[0])<=1
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
#     from keras.utils import np_utils
except ImportError:
    print("\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders")
    pass


#################### LONG SHORT TERM MEMORY (LSTM) DECODER ##########################

class LSTMRegression(object):

    """
    Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose


    def fit(self,X_train,y_train):

        """
        Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        #Add recurrent layer
        if keras_v1:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout)) #Within recurrent layer, include dropout
        else:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout)) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        if keras_v1:
            model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        else:
            model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted

yfull = np.load(f'rawData/Nh{Nh}/stimseed{stimseed}/stim.npy')

try:
    os.mkdir('LSTMbayesdecodeDataShuffle')
except FileExistsError:
    pass

try:
    os.mkdir(f'LSTMbayesdecodeDataShuffle/Nh{Nh}')
except FileExistsError:
    pass

try:
    os.mkdir(f'LSTMbayesdecodeDataShuffle/Nh{Nh}/stimseed{stimseed}')
except FileExistsError:
    pass

try:
    os.mkdir(f'LSTMbayesdecodeDataShuffle/Nh{Nh}/stimseed{stimseed}/modelseed{modelseed}')
except FileExistsError:
    pass

savepath = f'LSTMbayesdecodeDataShuffle/Nh{Nh}/stimseed{stimseed}/modelseed{modelseed}/deltat{deltat}'
if not os.path.exists(savepath):
    os.mkdir(savepath)

alpha = 0.5

num_steps = 1000
tf = num_steps
dt = 1
tvec = np.arange(0,tf,dt)
tf = num_steps
Nin = 100


pcon = 0.3

Nout = 100
Ns = [Nin, Nh, Nout]
layers = ['in','h','out']
numLayers = len(layers)
N_nrnVec = Ns
T_R = 50
delay = 0
delta_t = 2
train_frac = 0.5
slideVec = np.arange(0,tf-T_R+dt,dt)
tag = 'post'

print(f'deltat = {deltat}')
    
for li in range(numLayers):
    layer = layers[li]
    print(f'decoding layer {layer}')

    loadpath = f'rawData/Nh{Nh}/stimseed{stimseed}/modelseed{modelseed}/{layer}_spks.npy'
    spk = np.load(loadpath)
 
    tf=spk.shape[0]
    n_neurons=spk.shape[1]
    T=50
    n_time_bins=int(T/deltat)
    n_samples=tf-T+1
    y=yfull[:n_samples]
    #y4=y4full[:n_samples]
    #y20=y20full[:n_samples]
    X=np.zeros((n_samples,n_time_bins,n_neurons))
    for n in range(n_samples):
        sub=spk[n:n+T,:]
        s=[np.sum(sub[x:x+deltat,:],axis=0) for x in range(0,T,deltat)]
        X[n,:,:]=s

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.5,shuffle=shuffle)
    X_valid, X_test, y_valid, y_test = train_test_split(X_validate, y_validate, test_size=0.5,shuffle=shuffle)
    #X_train, X_validate, y4_train, y4_validate = train_test_split(X, y4, train_size=0.5,shuffle=False)
    #X_valid, X_test, y4_valid, y4_test = train_test_split(X_validate, y4_validate, test_size=0.5,shuffle=False)
    #X_train, X_validate, y20_train, y20_validate = train_test_split(X, y20, train_size=0.5,shuffle=False)
    #X_valid, X_test, y20_valid, y20_test = train_test_split(X_validate, y20_validate, test_size=0.5,shuffle=False)

    savepath_ytrain = savepath + f'/ytrain_{layer}.npy'
    np.save(savepath_ytrain,y_train)
    savepath_ytest = savepath + f'/ytest_{layer}.npy'
    np.save(savepath_ytest,y_test)

################################# DECODE SUM OF SINES #######################################################
    ### Get hyperparameters using Bayesian optimization based on validation set R2 values###

    #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
    #as a function of the hyperparameter we are fitting
    def rnn_evaluate(num_units,frac_dropout,n_epochs):
        num_units=int(num_units)
        frac_dropout=float(frac_dropout)
        n_epochs=int(n_epochs)
        model_rnn=LSTMRegression(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
        model_rnn.fit(X_train,y_train)
        y_valid_predicted_rnn=model_rnn.predict(X_valid)
        return r2_score(y_valid,y_valid_predicted_rnn)

    #Do bayesian optimization
    rnnBO = BayesianOptimization(f=rnn_evaluate, 
                                 pbounds={'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)},
                                 verbose=0)
    rnnBO.set_gp_params(alpha=1e-3)
    rnnBO.maximize()
    best_params=rnnBO.max['params']
    frac_dropout=float(best_params['frac_dropout'])
    n_epochs=int(best_params['n_epochs'])
    num_units=int(best_params['num_units'])

    # Run model w/ above hyperparameters
    model_rnn=LSTMRegression(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
    model_rnn.fit(X_train,y_train)
    ytrain_pred=model_rnn.predict(X_train) 
    savepath_ytrain_pred = savepath + f'/ytrain_pred_{layer}.npy'
    np.save(savepath_ytrain_pred,ytrain_pred)
    ytest_pred=model_rnn.predict(X_test)
    savepath_ytest_pred = savepath + f'/ytest_pred_{layer}.npy'
    np.save(savepath_ytest_pred,ytest_pred)
#############################################################################################################



################################## DECODE 4 HZ COMPONENT #######################################################
#    ### Get hyperparameters using Bayesian optimization based on validation set R2 values###
#
#    #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
#    #as a function of the hyperparameter we are fitting
#    def rnn_evaluate(num_units,frac_dropout,n_epochs):
#        num_units=int(num_units)
#        frac_dropout=float(frac_dropout)
#        n_epochs=int(n_epochs)
#        model_rnn=LSTMRegression(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
#        model_rnn.fit(X_train,y4_train)
#        y_valid_predicted_rnn=model_rnn.predict(X_valid)
#        return r2_score(y4_valid,y_valid_predicted_rnn)
#
#    #Do bayesian optimization
#    rnnBO = BayesianOptimization(f=rnn_evaluate, 
#                                 pbounds={'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)},
#                                 verbose=0)
#    rnnBO.set_gp_params(alpha=1e-3)
#    rnnBO.maximize()
#    best_params=rnnBO.max['params']
#    frac_dropout=float(best_params['frac_dropout'])
#    n_epochs=int(best_params['n_epochs'])
#    num_units=int(best_params['num_units'])
#
#    # Run model w/ above hyperparameters
#    model_rnn=LSTMRegression(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
#    model_rnn.fit(X_train,y4_train)
#    y4_pred=model_rnn.predict(X_test)
#    savepath_ypred = savepath + f'/y4pred_{layer}.npy'
#    np.save(savepath_ypred,y4_pred)
##############################################################################################################
#
#
#
################################## DECODE 20 HZ COMPONENT #######################################################
#    ### Get hyperparameters using Bayesian optimization based on validation set R2 values###
#
#    #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
#    #as a function of the hyperparameter we are fitting
#    def rnn_evaluate(num_units,frac_dropout,n_epochs):
#        num_units=int(num_units)
#        frac_dropout=float(frac_dropout)
#        n_epochs=int(n_epochs)
#        model_rnn=LSTMRegression(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
#        model_rnn.fit(X_train,y20_train)
#        y_valid_predicted_rnn=model_rnn.predict(X_valid)
#        return r2_score(y20_valid,y_valid_predicted_rnn)
#
#    #Do bayesian optimization
#    rnnBO = BayesianOptimization(f=rnn_evaluate, 
#                                 pbounds={'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)},
#                                 verbose=0)
#    rnnBO.set_gp_params(alpha=1e-3)
#    rnnBO.maximize()
#    best_params=rnnBO.max['params']
#    frac_dropout=float(best_params['frac_dropout'])
#    n_epochs=int(best_params['n_epochs'])
#    num_units=int(best_params['num_units'])
#
#    # Run model w/ above hyperparameters
#    model_rnn=LSTMRegression(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
#    model_rnn.fit(X_train,y20_train)
#    y20_pred=model_rnn.predict(X_test)
#    savepath_ypred = savepath + f'/y20pred_{layer}.npy'
#    np.save(savepath_ypred,y20_pred)
##############################################################################################################
