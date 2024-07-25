from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import os
import sys
# import keras

try:
    import keras
    keras_v1=int(keras.__version__[0])<=1
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
#     from keras.utils import np_utils
except ImportError:
    print("\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders")
    pass