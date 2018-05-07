import sys
import os
from os import path
lib_path = os.path.dirname(os.getcwd())
sys.path.append(lib_path)
from preprocess.PreProcess import PreProcess, load_data
from util.resampling import binary_downsampling, binary_upsampling
from util.metric import estimate_metric
from trainers.mlp_benchmark import MLP

import pandas as pd
import numpy as np
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# load train and validate data
print 'Loading data ...... \n'
data = pd.read_hdf('../data/train_bootstrap/train_bootstrap_0.hdf', 'bootstrap')
feature_column = range(data.shape[1] - 2)
X, y, w = data.loc[:,feature_column].values, data.loc[:,'y'].values, data.loc[:,'w'].values
w = np.zeros(len(y)) + 1
X_train, y_train, w_train = binary_upsampling(X, y, w)
Y_train = np.array([y_train, -(y_train-1)]).T

validate_data = pd.read_hdf('../data/validate_bootstrap/validate_bootstrap_0.hdf', 'bootstrap')
X_validate, y_validate, w_validate = validate_data.loc[:,feature_column], validate_data.loc[:,'y'], validate_data.loc[:,'w']
Y_validate = np.array([y_validate, -(y_validate-1)]).T


# grid search
print 'Grid search ...... \n'
learning_rate = 0.001
#hidden_length = [2, 3, 4, 5, 6, 7, 8]
#hidden_node_num = [4, 8, 12, 16, 20, 24, 28, 32]
hidden_length = [4, 5, 6, 7, 8, 10]
hidden_node_num = [5, 10, 20, 30,40, 50, 60]
for l_ in hidden_length:
    for n_ in hidden_node_num:
        # construct and train model
        model_name_ = 'mlp_lr001_depth{0}_num{1}_dropoutF_he_0504'.format(l_, n_)
        hidden_list = [n_] * l_
        config = {'model_name': model_name_,
                  'learning_rate': learning_rate,
                  'training_epochs': 10,
                  'batch_size': 32,
                  'architecture': hidden_list,
                  'weight_init': 'xavier',
                  'bias_init': 'xavier',
                  'early_stopping': 1000}
        print model_name_
        print config
        mlp = MLP(X_train.shape[1], hidden_list, 2, config)
        mlp.train(X_train, Y_train, X_validate, Y_validate)
