from collections import Counter, defaultdict
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from Utils.Dictionary import aggregation

SEED = 123 #used to help randomly select the data points


def get_train_test_split(outcome_col, grouping_col):
    y_distr = (get_distribution_counts(outcome_col))

    all_groups = set(grouping_col)
    batch_size = len(grouping_col)/len(all_groups)

    groups_y_df = pd.DataFrame()
    groups_y_df["groups"] = grouping_col
    groups_y_df["y"] = outcome_col
    groups_y_df.reset_index()

    groups_y_1 = set(groups_y_df.loc[groups_y_df['y']==1,'groups'])

    training_groups_1 = random.sample(groups_y_1,int(len(groups_y_1)/2))
    testing_groups_1 = groups_y_1 - set(training_groups_1)

    testing_df_1 = groups_y_df.loc[groups_y_df['groups'].isin(testing_groups_1),:]
    testing_groups_1 = set(testing_df_1['groups'])
    all_groups_zeros = all_groups - set(testing_df_1["groups"])

    testing_groups_0 = random.sample(all_groups_zeros,len(testing_groups_1))

    training_validation_groups = [x for x in all_groups_zeros if x not in testing_groups_0]

    training_groups_0 = random.sample(training_validation_groups,int(len(training_validation_groups)*0.8))
    validation_groups = set(training_validation_groups) - set(training_groups_0)

    print(" IN SPLIT:  LENGTH OF TRAINING GROUP FULL: ", len(training_groups_0+training_groups_1))
    train_indices_0 = [i for i, g in enumerate(grouping_col) if g in training_groups_0]
    train_indices_1 = [i for i, g in enumerate(grouping_col) if g in training_groups_1]
    training_indices_full = list(train_indices_0 + train_indices_1)
    validation_indices = [i for i, g in enumerate(grouping_col) if g in validation_groups]
    testing_indices_0 = [i for i, g in enumerate(grouping_col) if (g in testing_groups_0)]
    testing_indices_1 = [i for i, g in enumerate(grouping_col) if (g in testing_groups_1)]
    testing_indices_full = list(testing_indices_0 + testing_indices_1)
    return train_indices_0, train_indices_1, training_indices_full, validation_indices, testing_indices_0, testing_indices_1, testing_indices_full

def stratified_group_k_fold_original (y, groups, k, seed=None) :
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda : np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups) :
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda : np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold ( y_counts, fold ) :
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num) :
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x : -np.std(x[1])) :
        best_fold = None
        min_eval = None
        for i in range(k) :
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval :
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k) :
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def get_distribution ( y_vals ) :
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())
    return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]


def get_distribution_counts( y_vals ) :
    y_distr = Counter(y_vals)
    return [y_distr[i] for i in range(np.max(y_vals) + 1)]


def get_distribution_scalars( y_vals ) :
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())

    return [y_distr[i]/ y_vals_sum for i in range(np.max(y_vals) + 1)]

def get_distribution_percentages ( y_vals ) :
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())
    return [(y_distr[i] / y_vals_sum) for i in range(np.max(y_vals) + 1)]

def generate_balanced_arrays(df, x_features, outcome, grouping, no_groups):
 df = df[:,not (df[grouping].isin(no_groups))]
 y_test = (df[outcome]).to_numpy()
 X_test = df[x_features].to_numpy()

 while True:
  positive = np.where(y_test==1)[0].tolist()
  negative = np.random.choice(np.where(y_test==0)[0].tolist(),size = len(positive), replace = False)
  balance = np.concatenate((positive, negative), axis=0)
  np.random.shuffle(balance)
  input = X_test.iloc[balance, :]
  target = y_test.iloc[balance]
  yield input, target

def class_weights(y):
    total = len(y)
    neg = np.count_nonzero(y == 0)
    pos = np.count_nonzero(y == 1)
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0 : weight_for_0, 1 : weight_for_1}

    return class_weight


def class_counts(y):
    neg = np.count_nonzero(y == 0)
    pos = np.count_nonzero(y == 1)
    class_weight = {0 : neg, 1 : pos}
    return class_weight



def aggregate_outcomes(y, lookback):
    aggregated_y = []
    num_aggregate_samples = int(len(y)/lookback)

    for i in range(0, num_aggregate_samples):
        starting_pos = i*lookback
        ending_pos = i*lookback+lookback
        new_batch = y[starting_pos :ending_pos]
        aggregated_y.append(new_batch[0])

    print(" lENGTH OF AGGREGATED Y IS: ", len(aggregated_y))
    return aggregated_y

def aggregate_static_features(X, static_features, lookback):
    agg_df = pd.DataFrame()
    final_df = pd.DataFrame()
    num_aggregate_samples = int(X.shape[0]/lookback)

    for i in range(0, num_aggregate_samples):
        starting_pos = i*lookback
        ending_pos = i*lookback+lookback

        new_batch = X.iloc[starting_pos :ending_pos, :]
        batch_dict = {}
        for col in static_features:
            new_col = new_batch.loc[:, col]
            new_col1 = new_col.iloc[0]
            agg_df[col] = new_col1
            batch_dict.update({col: new_col1})
        final_df = final_df.append(batch_dict, ignore_index=True)
    final_df.to_csv("aggreagte.csv", index=False)
    return final_df

def generate_aggregates(X, dynamic_columns, lookback):
    agg_df = pd.DataFrame()
    final_df = pd.DataFrame()
    num_aggregate_samples = int(X.shape[0]/lookback)

    for i in range(0, num_aggregate_samples):
        starting_pos = i*lookback
        ending_pos = i*lookback+lookback

        new_batch = X.iloc[starting_pos :ending_pos, :]
        batch_dict = {}
        for col in dynamic_columns:
            new_col = new_batch.loc[:, col]
            col_aggregate = aggregation[col]
            if col_aggregate =='min':
                new_col1 = new_col.min()
                label="_min"
                agg_df[col + label] = new_col1
                batch_dict.update({col+label: new_col1})
            elif col_aggregate =='max':
                new_col1 = new_col.max()
                label = "_max"
                agg_df[col + label] = new_col1
                batch_dict.update({col+label: new_col1})
            elif col_aggregate=='min/max':
                new_col1 = new_col.min()
                label = "_min"
                agg_df[col + label] = new_col1
                batch_dict.update({col+label: new_col1})

                new_col2 = new_col.max()
                label = "_max"
                agg_df[col + label] = new_col2
                batch_dict.update({col+label: new_col2})


            else:
                new_col1 = new_col.mean()
                label = "_mean"
                agg_df[col + label] = new_col1
                batch_dict.update({col+label: new_col1})

        final_df = final_df.append(batch_dict, ignore_index=True)
    final_df.to_csv("aggreagte.csv", index=False)
    return final_df

def apply_func(df, col):
    return df.apply(aggregation[col], axis=1)

def impute(df, impute_columns):

    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(df[impute_columns])
    df[impute_columns] = imp.transform(df[impute_columns])

    return df[impute_columns]

def scale(df, scale_columns):

    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df[scale_columns]))
    normalized_df.columns = scale_columns

    print(" in scaling, columns are:", scale_columns, len(scale_columns))
    return normalized_df

def smote(X, y):
    over = SMOTE(sampling_strategy=0.9)
    under = RandomUnderSampler(sampling_strategy=0.9)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)
    return X, y