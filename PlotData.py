import os
import json

from sklearn.preprocessing import MinMaxScaler

from ProcessResults.ClassificationReport import ClassificationReport
from Models.LSTMAutoEncoder.Utils import process_data
from Utils.Data import impute, impute_df
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pylab import rcParams
import numpy as np
np.seterr(divide='ignore')

from numpy.random import seed

from Models.UtilsEqualDistributions import get_train_test_split, generate_aggregates

seed(7)

import seaborn as sns
import matplotlib.pyplot as plt

rcParams['figure.figsize'] = 8, 6
LABELS = ["0", "1"]



def main () :
    configs = json.load(open('Configuration.json', 'r'))
    epochs = configs['training']['epochs']
    grouping = configs['data']['grouping']
    dynamic_features = configs['data']['dynamic_columns']
    static_features = configs['data']['static_columns']

    outcomes = configs['data']['classification_outcome']
    lookback = configs['data']['batch_size']
    timeseries_path = configs['paths']['data_path']
    autoencoder_models_path = configs['paths']['autoencoder_models_path']
    test_data_path = configs['paths']['test_data_path']

    ##read, impute and scale dataset
    non_smotedtime_series = pd.read_csv(timeseries_path + "TimeSeriesAggregatedUpto0NonCum.csv")
    #non_smotedtime_series[dynamic_features] = impute(non_smotedtime_series, dynamic_features)
    normalized_timeseries = non_smotedtime_series.loc[:,dynamic_features]

    #intialise classification report which will house results of all outcomes
    classification_report = ClassificationReport()

    #save lstm performance for comparison with final outcome
    lstm_praucs = []
    ##start working per outcome
    for outcome in outcomes :
        train_indices_0, train_indices_1, training_indices_full,\
        validation_indices, testing_indices_0, testing_indices_1, testing_indices_full=\
            get_train_test_split(
            non_smotedtime_series[outcome].astype(int),
            non_smotedtime_series[grouping])




        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(normalized_timeseries.iloc[train_indices_0,:])
        normalized_timeseries.iloc[train_indices_0,:] = \
            scaler.transform(normalized_timeseries.iloc[train_indices_0,:])

        normalized_timeseries.iloc[validation_indices,:] = \
            scaler.transform(normalized_timeseries.iloc[validation_indices,:])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(normalized_timeseries.iloc[train_indices_1,:])
        normalized_timeseries.iloc[train_indices_1,:] = \
            scaler.transform(normalized_timeseries.iloc[train_indices_1,:])

        normalized_timeseries.iloc[train_indices_0,:] = \
            impute_df(normalized_timeseries.iloc[train_indices_0,:])
        normalized_timeseries.iloc[train_indices_1,:] = \
            impute_df(normalized_timeseries.iloc[train_indices_1,:])
        normalized_timeseries.iloc[validation_indices,:] = \
            impute_df(normalized_timeseries.iloc[validation_indices,:])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(normalized_timeseries.iloc[testing_indices_0,:])
        normalized_timeseries.iloc[testing_indices_0,:] = \
            scaler.transform(normalized_timeseries.iloc[testing_indices_0,:])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(normalized_timeseries.iloc[testing_indices_1,:])
        normalized_timeseries.iloc[testing_indices_1,:] = \
            scaler.transform(normalized_timeseries.iloc[testing_indices_1,:])


        normalized_timeseries.iloc[testing_indices_0,:] = \
            impute_df(normalized_timeseries.iloc[testing_indices_0,:])
        normalized_timeseries.iloc[testing_indices_1,:] = \
            impute_df(normalized_timeseries.iloc[testing_indices_1,:])


        normalized_timeseries.insert(0, grouping, non_smotedtime_series[grouping])
        normalized_timeseries.insert(0, outcome, non_smotedtime_series[outcome])

        zero_indices = train_indices_0
        one_indices = train_indices_1


        # plot
        if ("Mortality3D" in outcome) :

            for feature in range(0, len(normalized_timeseries.columns.values)):
                plt.figure()
                f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

                sns.histplot(normalized_timeseries.iloc[zero_indices,feature],ax=axes[0]).set_title("Zero " + str(normalized_timeseries.columns.values[feature]))
                sns.histplot(normalized_timeseries.iloc[one_indices,feature], ax=axes[1]).set_title(" One "+str(normalized_timeseries.columns.values[feature]))
                plt.savefig("Run/DataPlots/Train0VsTrain1"+outcome+str(feature)+".pdf")

        normalized_timeseries.drop(outcome, axis=1, inplace=True)

        X_train_0, X_valid, X_train_full, X_test_full,  y_train_full, y_test_full, timesteps, n_features= \
            process_data(normalized_timeseries, non_smotedtime_series, outcome, grouping, lookback,
                         train_indices_0, validation_indices, training_indices_full, testing_indices_full)

        normalized_timeseries.drop(grouping, axis=1, inplace=True)



if __name__ == '__main__' :
    main()
