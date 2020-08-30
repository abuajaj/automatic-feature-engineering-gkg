"""Python Extractor"""

from _csv import Error

import pandas as pd
from sklearn import preprocessing

import Utils
from config_dev import THRESHOLD
from db.DBManager import DBManager


class Extractor:
    """
    """

    def __init__(self, db_table):
        """Initializing
        """
        self.db_manager = DBManager()
        self.db_table = db_table
        self.size = 0
        self.features = pd.DataFrame()
        self.extracted_data = dict()
        self.common_keys = dict()

    def extract_features(self, lookup):
        """
        Extract new features from DB.
        Mapping each feature with his data (from DB).
        Handling missing data.
        Clean (if choose it) sparse features by THRESHOLD
        Normalize features data (LabelEncoder)
        Close DB connection by DBManager
        :return:
        """
        self.build(lookup)
        self.extract()
        self.data_handling(clean_sparse=False)
        self.db_manager.db_close()

    def build(self, entities):
        """
        :param entities:
        :return:
        """
        try:
            self.size = len(entities)
            self.prepare_database()
            for entity in entities:
                if entity:
                    self.fetch(Utils.normalize_name(Utils.normalize_uri(entity)))
        except Error as e:
            print("Error while build features", e)

    def fetch(self, entity):
        """
        Fetch from database the entity data (if exists)
        :param entity:
        :return:
        """
        # return a list of items
        result = self.db_manager.db_fetch(self.db_table, entity)
        if result and result is not None and len(result) > 0:
            for item in result:
                if 'name' in item.keys() and item['name']:
                    self.extracted_data[item['name']] = item
                    if item['data']:
                        self.prepare_common_keys(item['data'])

    def prepare_common_keys(self, item):
        """
        :param item:
        :return:
        """
        for key in item.keys():
            if key in self.common_keys:
                self.common_keys[key] = self.common_keys[key] + 1
            else:
                self.common_keys[key] = 1

    def extract(self):
        """
        :return:
        """
        for feature in self.common_keys.keys():
            values = []
            for i in range(0, self.size):
                values.append(None)

            for i, name in enumerate(self.extracted_data):
                entity = self.extracted_data.get(name)
                if feature in entity['data'].keys():
                    values[i] = entity['data'][feature]

            self.features[feature] = values

    def data_handling(self, clean_sparse=True):
        """
        Clean sparse features (if it choose)
        Handling missing data
        Fit & Transform data (Normalize)
        :param clean_sparse:
        :return:
        """
        if clean_sparse:
            self.clean_sparse_features()

        le = preprocessing.LabelEncoder()

        for feature, feature_data in self.features.items():

            feature_type = Utils.find_feature_type(self.features[feature])

            values = Utils.init_values(feature_type, self.size)

            for i, value in enumerate(feature_data):
                if feature_type == list:
                    values[i] = 0 if value is None else len(value)
                else:
                    if type(value) == list:
                        value = value[0]
                    values[i] = "" if value is None else value.replace("\"", "")

            feature_type = Utils.find_feature_type(values)
            values = Utils.missing_data(feature_type, values)
            values = le.fit_transform(values)

            self.features[feature] = values

    def clean_sparse_features(self):
        """
        Clean sparse features by THRESHOLD
        :return:
        """
        # Common features sorted
        var = {k: v for k, v in reversed(sorted(self.common_keys.items(), key=lambda item: item[1]))}

        for feature in self.features.keys():
            values_count = self.features[feature].value_counts()
            if round(len(values_count) / len(self.features[feature]), 2) < THRESHOLD:
                self.features.drop(feature, 1, inplace=True)

    def prepare_database(self):
        """
        Check if DB is connected, else establish a connection
        :return:
        """
        if self.db_manager and self.db_manager is not None and self.db_manager.is_connected():
            return True
        elif self.db_manager is not None and not self.db_manager.is_connected():
            self.db_manager.db_connect()
            return self.db_manager.is_connected()
        return False
