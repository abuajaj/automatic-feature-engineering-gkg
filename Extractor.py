"""Python Extractor"""
from _csv import Error

import numpy as np
import pandas as pd

import Utils
from config_dev import THRESHOLD
from DBManager import DBManager


class Extractor:
    """
    """

    def __init__(self, db_table, entity_key):
        """Initializing
        """
        self.db_manager = DBManager()
        self.db_table = db_table
        self.entity_key = entity_key
        self.df = None
        self.extracted_df = pd.DataFrame()
        self.extracted_data = dict()
        self.common_keys = dict()
        self.feature_importance_threshold = THRESHOLD

    def read(self, file):
        """
        Read entities from given dataset
        :param file:
        :return:
        """
        try:
            self.df = pd.read_csv(file, decimal=',')
            self.df = self.df.head(50)
        except Error as e:
            print("Error while reading file", e)

    def build(self):
        """
        :return:
        """
        try:
            self.prepare_database()
            # for index, row in self.df.iterrows():
            #     if self.entity_key in row.keys() and row[self.entity_key]:
            #         self.fetch(Utils.normalize_name(Utils.normalize_uri(row[self.entity_key])))
            for entity in self.df:
                if entity:
                    self.fetch(Utils.normalize_name(Utils.normalize_uri(entity)))

        except Error as e:
            print("Error while reading file", e)

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
        for key in item.keys():
            if key in self.common_keys:
                self.common_keys[key] = self.common_keys[key] + 1
            else:
                self.common_keys[key] = 1

    def extract(self):
        for feature in self.common_keys.keys():
            values = []
            for i in range(0, len(self.df)):
                values.append(None)

            for i, name in enumerate(self.extracted_data):
                entity = self.extracted_data.get(name)
                if feature in entity['data'].keys():
                    values[i] = entity['data'][feature]

            self.extracted_df[feature] = values

    def clean_and_normalize(self):

        self.clean_sparse_features()

        for feature, feature_data in self.extracted_df.items():

            feature_type = Utils.find_feature_type(self.extracted_df[feature])

            values = self.init_values(feature_type)

            for i, value in enumerate(feature_data):
                if feature_type == list:
                    values[i] = 0 if value is None else len(value)
                else:
                    if type(value) == list:
                        value = value[0]
                    values[i] = "" if value is None else value.replace("\"", "")

            values = Utils.missing_data(feature_type, values)

            self.extracted_df[feature] = values

    def clean_sparse_features(self):
        # Common features
        var = {k: v for k, v in reversed(sorted(self.common_keys.items(), key=lambda item: item[1]))}

        for feature in self.extracted_df.keys():
            values_count = self.extracted_df[feature].value_counts()
            if round(len(values_count) / len(self.extracted_df[feature]), 2) < self.feature_importance_threshold:
                self.extracted_df.drop(feature, 1, inplace=True)

    def prepare_database(self):
        """
        :return:
        """
        if self.db_manager and self.db_manager is not None and self.db_manager.is_connected():
            return True
        elif self.db_manager is not None and not self.db_manager.is_connected():
            self.db_manager.db_connect()
            return self.db_manager.is_connected()
        return False

    def init_values(self, feature_type):
        values = []
        for i in range(0, len(self.df)):
            if feature_type == str:
                values.append("")
            elif feature_type == object:
                values.append("")
            elif feature_type == list:
                values.append(0)
            else:
                values.append(np.nan)
        return values
