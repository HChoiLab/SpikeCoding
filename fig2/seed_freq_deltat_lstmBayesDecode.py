from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import os
import sys

Nh = int(sys.argv[1])
seed = int(sys.argv[2])
f = int(float(sys.argv[3]))
deltat = int(sys.argv[4])



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


yfull = np.load(f'rawData/stim_f{f}.npy')



try:
    os.mkdir('LSTMbayesdecodeData')
except FileExistsError:
    pass

try:
    os.mkdir(f'LSTMbayesdecodeData/f{f}')
except FileExistsError:
    pass

try:
    os.mkdir(f'LSTMbayesdecodeData/f{f}/Nh{Nh}')
except FileExistsError:
    pass

try:
    os.mkdir(f'LSTMbayesdecodeData/f{f}/Nh{Nh}/seed{seed}')
except FileExistsError:
    pass

savepath = f'LSTMbayesdecodeData/f{f}/Nh{Nh}/seed{seed}/deltat{deltat}'
try:
    os.mkdir(savepath)
except FileExistsError:
    pass
       
print(f'deltat = {deltat}')
layers=['in','h','out']; numLayers=len(layers)    
for li in range(numLayers):
    layer = layers[li]
    print(f'decoding layer {layer}')

    loadpath = f'rawData/Nh{Nh}/f{f}/seed{seed}/{layer}_spks.npy'
    spk = np.load(loadpath)
    tf=spk.shape[0]
    n_neurons=spk.shape[1]
    T=50
    n_time_bins=int(T/deltat)
    n_samples=tf-T+1
    y=yfull[:n_samples]
    X=np.zeros((n_samples,n_time_bins,n_neurons))
    for n in range(n_samples):
        sub=spk[n:n+T,:]
        s=[np.sum(sub[x:x+deltat,:],axis=0) for x in range(0,T,deltat)]
        X[n,:,:]=s

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.5,shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(X_validate, y_validate, test_size=0.5,shuffle=False)

    if li == 0: # save ytest only once per deltat
        savepath_ytest = savepath + '/ytest.npy'
        np.save(savepath_ytest,y_test)

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
#     rnnBO.maximize(init_points=20, n_iter=20, kappa=10)
#     print(f'rnnBO.max={rnnBO.max}')
# rnnBO.max={'target': 0.8978133107019641, 'params': {'frac_dropout': 0.09011958543363802, 'n_epochs': 20.520111647003624, 'num_units': 320.39851369495506}}
    best_params=rnnBO.max['params']
    frac_dropout=float(best_params['frac_dropout'])
    n_epochs=int(best_params['n_epochs'])
    num_units=int(best_params['num_units'])

    # Run model w/ above hyperparameters

    model_rnn=LSTMRegression(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
    model_rnn.fit(X_train,y_train)
    y_pred=model_rnn.predict(X_test)
#     y_test_predicted_rnn=model_rnn.predict(X_test)
#     mean_r2_rnn[i]=np.mean(get_R2(y_test,y_test_predicted_rnn))    
    #Print R2 values on test set
#     R2s_rnn=get_R2(y_test,y_test_predicted_rnn)
#     print('R2s:', R2s_rnn)
    #Add predictions of training/validation/testing to lists (for saving)           
#     y_pred_rnn_all.append(y_test_predicted_rnn)
#     y_train_pred_rnn_all.append(model_rnn.predict(X_train))
#     y_valid_pred_rnn_all.append(model_rnn.predict(X_valid))


#     rnn=SimpleRNNRegression()
#     rnn.fit(X_train,y_train)
#     y_pred=rnn.predict(X_test)

    savepath_ypred = savepath + f'/ypred_{layer}.npy'
    np.save(savepath_ypred,y_pred)
        
