import os
import json

from sklearn.metrics import auc

from Models.LSTMAutoEncoder.LSTMAutoEncoder import LSTMAutoEncoder
from Models.LSTMAutoEncoder.Utils import process_data, lstm_flatten
from ProcessResults.ClassificationReport import ClassificationReport
from Utils.Data import flatten, scale, impute
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pylab import rcParams
from Models.Utils import class_weights, class_counts, aggregate_static_features, aggregate_outcomes
import numpy as np
np.seterr(divide='ignore')

from Models.Constraints import WeightsOrthogonalityConstraint, UncorrelatedFeaturesConstraint, AttentionDecoder
from keras.models import Model, load_model

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
    non_smotedtime_series = pd.read_csv(timeseries_path + "TimeSeriesAggregatedUpto0NonCum.csv")
    non_smotedtime_series[dynamic_features] = impute(non_smotedtime_series, dynamic_features)
    normalized_timeseries = scale(non_smotedtime_series, dynamic_features)
    normalized_timeseries.insert(0, grouping, non_smotedtime_series[grouping])

    #intialise classification report which will house results of all outcomes
    classification_report = ClassificationReport()

    #save lstm performance for comparison with final outcome
    lstm_praucs = []
    ##start working per outcome
    for outcome in outcomes :
        train_indices_0, validation_indices, training_indices_full, testing_indices_full=\
            get_train_test_split(
            non_smotedtime_series[outcome].astype(int),
            non_smotedtime_series[grouping])

        ##Load LSTM models if they exist, otherwise train new models and save them
        autoencoder_filename = autoencoder_models_path + configs['model']['name'] + outcome + '.h5'

        X_train_0, X_valid, X_train_full, X_test_full,  y_train_full, y_test_full, timesteps, n_features= \
            process_data(normalized_timeseries, non_smotedtime_series, outcome, grouping, lookback,
                         train_indices_0, validation_indices, training_indices_full, testing_indices_full)
        if ("ZINA" not in outcome) :
            if os.path.isfile(autoencoder_filename):
                print(" Autoencoder trained model exists for oucome", outcome,"file:" , autoencoder_filename)
                autoencoder = LSTMAutoEncoder(configs['model']['name'] + outcome, outcome,
                                              timesteps, n_features,
                                              custom_objects={
                                                  'WeightsOrthogonalityConstraint' : WeightsOrthogonalityConstraint,
                                                  'UncorrelatedFeaturesConstraint' : UncorrelatedFeaturesConstraint,
                                                  'AttentionDecoder' : AttentionDecoder},
                                              saved_model = autoencoder_filename)


                autoencoder.summary()



            else :
                print("Autencoder trained model does not exist for outcome", outcome, "file:", autoencoder_filename)
                autoencoder = LSTMAutoEncoder(configs['model']['name'] + outcome, outcome,
                                              timesteps, n_features,
                                              custom_objects={
                                                  'WeightsOrthogonalityConstraint' : WeightsOrthogonalityConstraint,
                                                  'UncorrelatedFeaturesConstraint' : UncorrelatedFeaturesConstraint,
                                                  'AttentionDecoder' : AttentionDecoder}
                                              )
                autoencoder.summary()

                autoencoder.fit(X_train_0, epochs, lookback, X_valid, 2)
                autoencoder.plot_history()


            X_train_predictions= autoencoder.predict(X_train_full)

            print(" DIM OF X TRAIN PREDICTIONS: ", X_train_predictions.shape)
            mse_train = np.mean(np.power(lstm_flatten(X_train_full) - lstm_flatten(X_train_predictions), 2), axis=1)
            print(" DIM OF MSE TRAIN: ", len(mse_train))

            test_x_predictions = autoencoder.predict(X_test_full)

            mse_test = np.mean(np.power(lstm_flatten(X_test_full) - lstm_flatten(test_x_predictions), 2), axis=1)

            aggregated_y_test = aggregate_outcomes( y_test_full, lookback)

            test_error_df = pd.DataFrame({'Reconstruction_error' : mse_test,
                                          'True_class' : aggregated_y_test})

            pred_y, best_threshold, precision_rt, recall_rt = \
                  autoencoder.predict_binary(test_error_df.True_class, test_error_df.Reconstruction_error)

            autoencoder.output_performance(test_error_df.True_class, pred_y)
            autoencoder.plot_reconstruction_error(test_error_df, best_threshold)
            autoencoder.plot_roc(test_error_df)
            autoencoder.plot_pr(precision_rt, recall_rt)
            lstm_prauc = auc(recall_rt, precision_rt)
            lstm_praucs.append(lstm_prauc)


            #Feature Selector

            X_train = (non_smotedtime_series.iloc[training_indices_full,:]).copy()
            X_train_aggregates = generate_aggregates ( X_train, dynamic_features, lookback ) #ZI CHANGE TO BATCH SIZE
            X_train_static = aggregate_static_features(X_train, static_features, lookback)
            X_train = pd.concat([X_train_aggregates, X_train_static.reindex(X_train_aggregates.index)], axis=1)
            X_train['mse'] = mse_train
            y_train = aggregate_outcomes( y_train_full, lookback)
            print(" DIM OF XGB AGGREGATE DF TRAIN: ", X_train.shape)
            print(" LEN OF XGB AGGREGATE DF TRAIN OUTCOME: ", len(y_train))


            X_test = (non_smotedtime_series.iloc[testing_indices_full, :]).copy()
            X_test_aggregates = generate_aggregates(X_test, dynamic_features, lookback)  # ZI CHANGE TO BATCH SIZE
            X_test_static = aggregate_static_features(X_test, static_features, lookback)
            X_test = pd.concat([X_test_aggregates, X_test_static.reindex(X_test_aggregates.index)], axis=1)
            X_test['mse'] = mse_test
            y_test = aggregate_outcomes( y_test_full, lookback)
            print(" DIM OF XGB AGGREGATE DF TEST: ", X_test.shape)
            print(" LEN OF XGB AGGREGATE DF TEST OUTCOME: ", len(y_test))

            ########

            X_test.to_csv("static_aggretate.csv", index=False)
            static_baseline_classifier = XGBoostClassifier(X_train,
                                                                  y_train, outcome, grouping)

            print(" checking train and test dims in xgboost: ", X_train.shape, len(y_train))
            static_baseline_classifier.fit("aggregate_static", mse_train*100)

            y_pred_binary, best_threshold, precision_rt, recall_rt, yhat = \
                static_baseline_classifier.predict(X_test, y_test)


            static_baseline_classifier.output_performance(y_test, y_pred_binary)
            static_baseline_classifier.plot_pr(precision_rt, recall_rt, "XGBoost Static")
            static_baseline_classifier.plot_feature_importance(X_test.columns)

            to_write_for_plotting = X_test
            to_write_for_plotting['outcome'] = y_test
            to_write_for_plotting.to_csv(test_data_path+outcome+".csv", index=False)

            #add to classification report

            classification_report.add_model_result(outcome,y_test, y_pred_binary, best_threshold,
                                                   precision_rt, recall_rt, yhat)


            if ("3D" not in outcome) :
                classification_report.plot_distributions_vs_aucs()
                classification_report.plot_pr_auc()
                classification_report.plot_auc()
                classification_report.compare_lstim_xgboost(lstm_praucs)


if __name__ == '__main__' :
    main()
