import argparse
from _csv import Error
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import Utils
from config_dev import DATASET_DIR, LOOKUP, TARGET
from models.Evaluator import Evaluator
from models.Extractor import Extractor
from models.MetaFeaturesHandler import MetaFeaturesHandler

plt.style.use('seaborn-whitegrid')


class Main:
    """
    """

    def __init__(self) -> object:
        """Initializing
        """
        self.data_folder = Path(DATASET_DIR)

    def read_dataset(self, dataset_file, dataset_size=-1):
        """
        Read entities from given dataset
        :param dataset_file:
        :param dataset_size: Or part of dataset ['all', Integers]
        :return:
        """

        try:
            dataset = pd.read_csv(self.data_folder / dataset_file, decimal=',')
            classes = list(set(dataset[TARGET]))
            if dataset_size > -1:
                dataset = dataset.head(int(dataset_size))

            return dataset, classes

        except Error as e:
            print("Error while reading file", e)

    def build_meta_features(self, dataset_file, db_table_name, dataset_size=-1):
        """
        :param dataset_file:
        :param db_table_name:
        :param dataset_size:
        :return:
        """
        # init & read
        dataset, classes = self.read_dataset(dataset_file, dataset_size)

        # extract new features
        extractor = Extractor(db_table_name)
        extractor.extract_features(dataset[LOOKUP])
        features = extractor.features
        labels = dataset[TARGET]

        # build meta features
        meta_features_handler = MetaFeaturesHandler()
        meta_features = meta_features_handler.build(features, labels)

        return features, meta_features, dataset, classes

    def read_meta_features(self, dataset_file, dataset_size=-1):
        """
        :param dataset_file:
        :param dataset_size:
        :return:
        """

        try:
            stored_meta_features = "output/" + dataset_file
            dataset = pd.read_csv(self.data_folder / stored_meta_features, decimal=',')
            if dataset_size > -1:
                dataset = dataset.head(int(dataset_size))

            return dataset

        except Error as e:
            print("Error while reading file", e)

    def build(self, train_file_1, train_file_2, test_file, train_table_1, train_table_2, test_table):
        """

        :param train_file_1:
        :param train_file_2:
        :param test_file:
        :param train_table_1:
        :param train_table_2:
        :param test_table:
        :return:
        """
        _evaluator = Evaluator()
        # build
        features, meta_features, dataset, classes = self.build_meta_features(train_file_1, train_table_1)

        # evaluate X features
        meta_features_X = _evaluator.evaluate(features, meta_features, dataset, classes)
        stored_meta_features = "output/" + train_file_1
        Utils.write_dataset(meta_features_X, self.data_folder / stored_meta_features)

        # build
        features, meta_features, dataset, classes = self.build_meta_features(train_file_2, train_table_2)

        # evaluate Y features
        meta_features_Y = _evaluator.evaluate(features, meta_features, dataset, classes)
        stored_meta_features = "output/" + train_file_2
        Utils.write_dataset(meta_features_Y, self.data_folder / stored_meta_features)

        # build
        features, meta_features, dataset, classes = self.build_meta_features(test_file, test_table)

        # evaluate Y features
        meta_features_Z = _evaluator.evaluate(features, meta_features, dataset, classes)
        stored_meta_features = "output/" + test_file
        Utils.write_dataset(meta_features_Z, self.data_folder / stored_meta_features)

    def test(self, train_file_1, train_file_2, test_file, train_table_1, train_table_2, test_table):
        """
        :param train_file_1:
        :param train_file_2:
        :param test_file:
        :param train_table_1:
        :param train_table_2:
        :param test_table:
        :return:
        """
        _evaluator = Evaluator()
        # X
        meta_features_X = self.read_meta_features(train_file_1)
        X = meta_features_X.drop([LOOKUP, TARGET], axis=1)
        for key in X:
            X[key] = Utils.missing_data(float, X[key])
        y = meta_features_X[TARGET]
        # fit a model with X
        _evaluator.model.fit(X, y)

        # Y
        meta_features_Y = self.read_meta_features(train_file_2)
        X = meta_features_Y.drop([LOOKUP, TARGET], axis=1)
        for key in X:
            X[key] = Utils.missing_data(float, X[key])
        y = meta_features_Y[TARGET]
        # fit a model with Y
        _evaluator.model.fit(X, y)

        # predict Z
        meta_features_Z = self.read_meta_features(test_file)
        X_test = meta_features_Z.drop([LOOKUP, TARGET], axis=1)
        for key in X_test:
            X_test[key] = Utils.missing_data(float, X_test[key])
        y_pred = _evaluator.predict(X_test)
        print(y_pred)

        print(mean_squared_error(meta_features_Z[TARGET], y_pred))
        print(mean_absolute_error(meta_features_Z[TARGET], y_pred))

        n_features = Utils.get_n_features(meta_features_Z)
        print("THE FINAL RESULT:")
        print(n_features)


def main(method, train_file_1, train_file_2, test_file, train_table_1, train_table_2, test_table):
    """
    Build or Test
    :param method:
    :param train_file_1:
    :param train_file_2:
    :param test_file:
    :param train_table_1:
    :param train_table_2:
    :param test_table:
    :return:
    """
    print('>>> Start.. ', Utils.print_time())

    # init
    _main = Main()

    if method == "build":
        "###################################### Builds ######################################"

        _main.build(train_file_1, train_file_2, test_file, train_table_1, train_table_2, test_table)

    if method == "test":
        "###################################### TEST ######################################"

        _main.test(train_file_1, train_file_2, test_file, train_table_1, train_table_2, test_table)

    print('<<< Stop.. ', Utils.print_time())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('method', help='Build or Test')
    parser.add_argument('train_file_1', help='Path to the input data file')
    parser.add_argument('train_file_2', help='Path to the input data file')
    parser.add_argument('test_file', help='Path to the input data file')
    parser.add_argument('train_table_1', help='Path to the input data file')
    parser.add_argument('train_table_2', help='Path to the input data file')
    parser.add_argument('test_table', help='Path to the input data file')
    args = parser.parse_args()
    method = args.method
    train_file_1 = args.train_file_1
    train_file_2 = args.train_file_2
    test_file = args.test_file
    train_table_1 = args.train_table_1
    train_table_2 = args.train_table_2
    test_table = args.test_table

    print(method)
    print(train_file_1)
    print(train_file_2)
    print(test_file)
    print(train_table_1)
    print(train_table_2)
    print(test_table)
    main(method, train_file_1, train_file_2, test_file, train_table_1, train_table_2, test_table)
