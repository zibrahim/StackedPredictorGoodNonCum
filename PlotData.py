import os
import json

from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler

from Models.LSTMAutoEncoder.LSTMAutoEncoder import LSTMAutoEncoder
from Models.LSTMAutoEncoder.Utils import process_data, lstm_flatten
from ProcessResults.ClassificationReport import ClassificationReport
from Utils.Data import flatten, scale, impute, impute_df, scale_impute
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from Models.Utils import aggregate_static_features, aggregate_outcomes
import numpy as np
np.seterr(divide='ignore')

from Models.Constraints import WeightsOrthogonalityConstraint, UncorrelatedFeaturesConstraint, AttentionDecoder
from numpy.random import seed

from Models.Utils import get_train_test_split, generate_aggregates
from Models.XGBoost.XGBoost import XGBoostClassifier
import os.path

seed(7)

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
    non_smotedtime_series = pd.read_csv(timeseries_path + "TimeSeriesAggregatedUpto0.csv")
    #non_smotedtime_series[dynamic_features] = impute(non_smotedtime_series, dynamic_features)
    xgboost_series_original = non_smotedtime_series.loc[:,dynamic_features+static_features]
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

        normalized_timeseries = scale_impute(normalized_timeseries, train_indices_0, train_indices_1,
                                            validation_indices, testing_indices_0,
                                            testing_indices_1)

        normalized_timeseries.insert(0, grouping, non_smotedtime_series[grouping])

        normalized_timeseries.insert(0, outcome, non_smotedtime_series[outcome])

        zero_indices = [train_indices_0,testing_indices_0]
        one_indices = [train_indices_1, testing_indices_1]


        for feature in range(0, len(normalized_timeseries.columns.values)):
            plt.figure()
            f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

            sns.histplot(normalized_timeseries.iloc[zero_indices,feature],ax=axes[0]).set_title("Zero " + str(normalized_timeseries.columns.values[feature]))
            sns.histplot(normalized_timeseries.iloc[one_indices,feature], ax=axes[1]).set_title(" One "+str(normalized_timeseries.columns.values[feature]))
            plt.savefig("Run/DataPlots/Train0VsTrain1"+outcome+str(feature)+".pdf")

        normalized_timeseries.drop(outcome, axis=1, inplace=True)

        normalized_timeseries.drop(grouping, axis=1, inplace=True)



if __name__ == '__main__' :
    main()
