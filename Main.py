"""Python client calling Knowledge Graph Search API."""
import json
import re
import urllib
import urllib.request
from urllib.parse import urlencode

import pandas as pd

from Parser import Parser


class Main:
    """
    """

    def __init__(self):
        """Initializing
        """
        self.data = None
        self.api_key = open('.api_key').read()
        self.api_url = 'https://kgsearch.googleapis.com/v1/entities:search'
        self.api_params = {
            'query':  '',
            'limit':  10,
            'indent': True,
            'key':    self.api_key,
            # 'types': ['Person'],
        }
        self.parser = Parser()

    def build_query(self, query):
        """ Build HTTP query and send to GKG API,
            As response; return graph as json-LD object
            :param: query
        """
        if query == '':
            return

        self.api_params['query'] = query
        url = self.api_url + '?' + urllib.parse.urlencode(self.api_params)
        # print(url)
        # Todo Handling errors
        response = json.loads(urllib.request.urlopen(url).read())
        return response

    def load_data(self, filename):
        """ Load dataset from file
            :param filename (string)
            :return DataFrame of data
        """
        try:
            self.data = pd.read_csv(filename)
        except FileNotFoundError:
            msg = filename + ": FileNotFoundError"
            e = Exception(msg)
            print(e)

    def extract_features(self, entity_column):
        """ Extract candidate features from built dataset using GKG API
            :param entity_column:
            :return result of candidate features
        """
        if entity_column == '' or entity_column is None:
            return {}

        entity_features = {}
        for i, search in enumerate(self.data[entity_column]):
            if search:
                # print("before: ", search)
                search = self.normalize_text(search)
                # print("after: ", search)
                graph_response = self.build_query(search)
                if graph_response:
                    features = self.parser.parse(graph_response)
                    entity_features[search] = features
                    self.add_to_dataset(features, i)

        return entity_features

    def normalize_text(self, text):
        """Returns a normalized string based on the specific string.
            You can add default parameters as you like (they should have default values!)

            :param text (str) the text to normalize

            :returns: string. the normalized text.

            lower casing, padding punctuation with white spaces
        """
        text = re.sub("_", " ", text)
        text = re.sub("[!@#$+%*:()',;}{{})(\\]\\[-]", "", text)
        text = text.lower().strip()

        return text

    def add_to_dataset(self, columns, index):
        for column, value in columns.items():
            if column not in self.data.keys():
                self.data[column] = pd.Series([])
            self.data.loc[index, column] = value


main = Main()
graph_response = main.build_query('The Color Purple')
features_list = main.parser.parse(graph_response)
print(features_list)

# parser = Parser()
# parser.parse(graph_response)

# main.load_data("datasets/dataset_189_baseball.csv")
# extracted_features = main.extract_features('Player')
# print(extracted_features)
# s = Selection()

# load the dataset
# X, y = s.prepare_dataset(main.data)
# # split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# # prepare input data
# X_train_enc, X_test_enc = s.prepare_inputs(X_train, X_test)
# # prepare output data
# y_train_enc, y_test_enc = s.prepare_targets(y_train, y_test)
# # feature selection
# X_train_fs, X_test_fs, fs = s.select_features(X_train_enc, y_train_enc, X_test_enc)
# # what are scores for the features
# for i in range(len(fs.scores_)):
#     print('Feature %d: %f' % (i, fs.scores_[i]))
#
# # plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()

# for element in graph_response['itemListElement']:
#     print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')
