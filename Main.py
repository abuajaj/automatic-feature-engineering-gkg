from __future__ import print_function

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import Utils
from config_dev import TOP_N_FEATURES
from Extractor import Extractor
from models.Evaluation import Evaluation

plt.style.use('seaborn-whitegrid')

data_folder = Path("/Users/mabuajaj/PycharmProjects/AFEGKG/datasets/")
lookup = 'lookup'
target = 'target'

scaler = MinMaxScaler(feature_range=(0, 1))
le = preprocessing.LabelEncoder()
evaluate = Evaluation()


def prepare_before(X, y):
    for x in X:
        feature_type = Utils.find_feature_type(X[x])
        if feature_type is str or feature_type is object:
            X[x] = le.fit_transform(Utils.missing_data(X[x].dtype, list(X[x])))
        else:
            X[x] = scaler.fit_transform(np.array(X[x]).reshape(-1, 1))

    evaluate.features_importance(X, y)
    scores = evaluate.model_1(X, y)
    print('Before >>> Scores (mean):')
    print(scores)
    return X, y


def prepare_after(X, y, table_name):
    print('Extractor >>> Start.. ', Utils.print_time())

    extractor = Extractor(table_name, lookup)
    extractor.df = X[lookup]
    extractor.build()

    extractor.extract()
    extractor.clean_and_normalize()

    X.drop([lookup], axis=1, inplace=True)
    if not X.empty:
        _X = pd.concat([X, extractor.extracted_df], axis=1, sort=False)
    else:
        _X = extractor.extracted_df

    for x in _X:
        feature_type = Utils.find_feature_type(_X[x])
        if feature_type is str or feature_type is object:
            _X[x] = le.fit_transform(Utils.missing_data(_X[x].dtype, list(_X[x])))
        else:
            _X[x] = scaler.fit_transform(np.array(_X[x]).reshape(-1, 1))

    features_meta_after = evaluate.features_importance(_X, y)

    extracted_features = features_meta_after['feature_imp'][extractor.extracted_df.keys()]
    extracted_features = Utils.sort_keys(extracted_features)
    not_extracted_features = list(extracted_features)[TOP_N_FEATURES:]
    extracted_features = list(extracted_features)[:TOP_N_FEATURES]
    _X.drop(not_extracted_features, 1, inplace=True)

    features_meta_after_clean = evaluate.features_importance(_X, y)

    scores = evaluate.model_1(_X, y)
    print('After >>> Scores (mean) with new features:')
    print(scores)

    print('Extractor >>> The End is near!')
    extractor.db_manager.db_close()


def main():
    print('>>> Start.. ', Utils.print_time())

    # dataset_file = data_folder / "movies_after_matching.csv"
    dataset_file = data_folder / 'dataset_189_baseball_after_matching.csv'
    table_name = 'baseballTBR'

    dataset = pd.read_csv(dataset_file, decimal=',')
    # dataset = dataset.head(100)
    y = dataset[target]

    # BEFORE
    X = dataset.drop([target, lookup], axis=1)
    if not X.empty:
        X, y = prepare_before(X, y)

    # AFTER
    X = dataset.drop([target], axis=1)
    prepare_after(X, y, table_name)

    print('<<< Stop.. ', Utils.print_time())


if __name__ == "__main__":
    main()
