import numpy as np
import random
import pandas as pd
from collections import Counter, defaultdict


def process_data(dynamic_series, full_series, outcome, grouping, lookback,
                 train_indices_0, validation_indices, training_indices_full, testing_indices_full):

    dynamic_series.insert(len(dynamic_series.columns), outcome, full_series[outcome])
    dynamic_series[outcome] = dynamic_series[outcome].astype(int)

    X_cols = (dynamic_series.columns).tolist()

    input_X = dynamic_series.loc[:,
              dynamic_series.columns.isin(X_cols)] # converts the df to a numpy array
    input_y = dynamic_series[outcome].values

    n_features = input_X[X_cols].shape[1] -2 # number of features

    X_train_0 = input_X.iloc[train_indices_0]
    X_valid = input_X.iloc[validation_indices]


    X_train_full = input_X.iloc[training_indices_full]
    y_train_full = input_y[training_indices_full]

    X_test_full = input_X.iloc[testing_indices_full]
    y_test_full = input_y[testing_indices_full]

    X_train_0 = X_train_0.drop(grouping, axis=1)
    X_train_0  =X_train_0.drop(outcome, axis=1)
    X_train_0 = X_train_0.to_numpy()
    X_train_0 = X_train_0.reshape(-1, lookback, n_features)

    X_valid = X_valid.drop(grouping, axis=1)
    X_valid = X_valid.drop(outcome, axis=1)
    X_valid = X_valid.to_numpy()
    X_valid = X_valid.reshape(-1, lookback, n_features)

    X_train_full = X_train_full.drop(grouping, axis=1)
    X_train_full  =X_train_full.drop(outcome, axis=1)
    X_train_full = X_train_full.to_numpy()
    X_train_full = X_train_full.reshape(-1, lookback, n_features)

    X_test_full = X_test_full.drop(grouping, axis=1)
    X_test_full  =X_test_full.drop(outcome, axis=1)
    X_test_full = X_test_full.to_numpy()
    X_test_full = X_test_full.reshape(-1, lookback, n_features)

    timesteps = X_train_0.shape[1]  # equal to the lookback
    n_features = X_train_0.shape[2]  # 59

    return X_train_0, X_valid, X_train_full, X_test_full, y_train_full, y_test_full, timesteps, n_features


def lstm_flatten ( X ) :
    '''
    Flatten a 3D array.

    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.

    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]) :
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)
