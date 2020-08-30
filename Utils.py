import json
import operator
import re
from _csv import Error
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from config_dev import NOT_ALLOWED_TYPES, TARGET, TOP_N_FEATURES


def normalize_uri(uri):
    """
    :param uri:
    :return:
    """
    if "<" in uri and ">" in uri:
        new_uri = uri[1:len(uri) - 1]
        new_uri = new_uri.replace("http://rdf.freebase.com/ns", "")
        new_uri = new_uri.replace("http://rdf.freebase.com/key", "")
        new_uri = new_uri.replace("/http://www.w3.org/2000/01/rdf-schema#", "")
        new_uri = new_uri.replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "")
        new_uri = new_uri.replace("http://www.w3.org/2000/01/rdf-schema#", "")
        new_uri = new_uri.replace("http://www.w3.org/2002/07/owl#", "")
        new_uri = new_uri.replace("http://dbpedia.org/resource/", "")
        new_uri = re.sub(r"\"\^\^<.*?XMLSchema#gYear", "", new_uri)
        new_uri = re.sub(r"\"\^\^<.*?XMLSchema#date", "", new_uri)
        new_uri = new_uri.replace(".", "/")
        return new_uri
    elif 'dbpedia' in uri:
        new_uri = uri.replace("http://dbpedia.org/resource/", "")
        return new_uri

    return uri


def normalize_name(name):
    """
    :param name:
    :return:
    """
    if name and name is not None:
        new_uri = name.replace("_", " ")
        return new_uri
    return name


def parse_triple(line):
    """
    Parse given FB line into three parts
    :param line:
    :return:
    """
    parts = line.split("\t")
    subject = normalize_uri(parts[0])
    predicate = normalize_uri(parts[1])
    Object = normalize_uri(parts[2])
    return subject, predicate, Object


def handle_duplicate(current_topic):
    """
    :param current_topic:
    :return:
    """

    if 'label' in current_topic.keys() and '/type/object/name' in current_topic.keys():
        current_topic.pop('/type/object/name')
    if 'type' in current_topic.keys() and '/type/object/type' in current_topic.keys():
        current_topic.pop('/type/object/type')

    for key in list(current_topic.keys()):
        if 1 == len(current_topic[key]):
            current_topic[key] = current_topic[key][0]
        if key in NOT_ALLOWED_TYPES or re.search('/wikipedia/', key):
            current_topic.pop(key)

    return current_topic


def handle_language(current_topic):
    """
    Handling lang features
    :param current_topic:
    :return:
    """
    for key in list(current_topic.keys()):
        if isinstance(current_topic[key], list):
            for lang in current_topic[key]:
                if re.search("@en", lang):
                    current_topic[key] = clean_lang(lang)
                    break

    if 'label' in current_topic.keys():
        if isinstance(current_topic['label'], list) and len(current_topic['label']) > 0:
            for label in current_topic['label']:
                if label and (not is_english(label) or not is_english2(label)):
                    return None
        elif not is_english(current_topic['label']) or not is_english2(current_topic['label']):
            return None

    return current_topic


def clean_lang(s):
    """
    Clean string from language declaration
    :param s:
    :return:
    """
    return s.replace("@en", "").replace("\"", "").replace("\\", "")


def is_english(s):
    """
    Check if given string is english string
    :param s:
    :return: boolean
    """
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def is_english2(s):
    """
    Check if given string is english string
    :param s:
    :return: boolean
    """
    if (re.search("\"@", s) or re.search("@", s)) and not re.search("\"@en", s):
        return False
    else:
        return True


def print_time():
    """
    Print curent time
    :return:
    """
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "Current Time = ", current_time


def print_json(data):
    """
    Print given object as JSON format
    :param data:
    :return:
    """
    json_formatted_str = json.dumps(data, indent=2)
    print(json_formatted_str)
    print()


def missing_data(feature_type, values):
    """
    Fill missing data according to data type
    :param feature_type:
    :param values:
    :return: data without missing data
    """

    if feature_type == int or feature_type == float:
        if feature_type == int:
            values = [0 if x == '' else int(x) for x in values]
        if feature_type == float:
            values = [.0 if x == '' else float(x) for x in values]

        for i in range(0, len(values)):
            if values[i] is np.nan:
                x = round(np.nanmean(values), 3)
                values[i] = x

    if feature_type == object or feature_type == str:
        reg = re.compile('^[\$\%\&\*\()\@\!\?\ \.\,\'\"]+$')
        for i in range(0, len(values)):
            if values[i] is None or reg.match(values[i]):
                values[i] = ""

        mode = max(set(values), key=values.count)
        for i in range(0, len(values)):
            if values[i] == '' or not values[i] or values[i] is None:
                values[i] = mode
    return values


def find_feature_type(column):
    """
    Find given data column' type
    :param column:
    :return: data type
    """
    column_types = dict()
    for value in column:
        if value:
            value_type = type(value)
            if value_type is str:
                try:
                    tmp = float(value)
                    if isinstance(tmp, int) and not isinstance(tmp, bool):
                        value_type = int
                    elif isinstance(tmp, (float, complex)) and not isinstance(tmp, bool):
                        value_type = float
                except ValueError:
                    return str
            if value_type in column_types:
                column_types[value_type] = column_types[value_type] + 1
            else:
                column_types[value_type] = 1
    return max(column_types.items(), key=operator.itemgetter(1))[0]


def sort_keys(items):
    """
    Sort given keys
    :param items:
    :return: sorted items
    """
    return {k: v for k, v in reversed(sorted(items.items(), key=lambda item: item[1]))}


def init_values(feature_type, length):
    """
    Init values by given data type
    :param feature_type:
    :param length:
    :return:
    """
    values = []
    for i in range(0, length):
        if feature_type == str:
            values.append("")
        elif feature_type == object:
            values.append("")
        elif feature_type == list:
            values.append(0)
        else:
            values.append(np.nan)
    return values


def split_dataset(dataset, lookup, target, test_size=.2):
    """
    Split dataset by 'test_size'
    :param test_size: Train & Test percents
    :param dataset:
    :param lookup:
    :param target:
    :param test_size:
    :return: X_train, X_test, y_train, y_test
    """
    X = dataset.drop([target, lookup], axis=1)
    y = dataset[target]

    return train_test_split(X, y, test_size=test_size, random_state=0)


def normalize(feature_data):
    """
    :param feature_data:
    :return:
    """
    oe = preprocessing.LabelEncoder()

    feature_type = find_feature_type(feature_data)
    values = init_values(feature_type, len(feature_data))

    for i, value in enumerate(feature_data):
        if feature_type == list:
            values[i] = 0 if value is None else len(value)
        else:
            if type(value) == list:
                value = value[0]
            values[i] = "" if value is None else value

    feature_type = find_feature_type(values)
    values = missing_data(feature_type, values)
    values = oe.fit_transform(values)
    return values


def write_dataset(dataset, file_name):
    """
    :return:
    """
    try:
        if isinstance(dataset, pd.DataFrame):
            dataset.to_csv(file_name, sep=',', encoding='utf-8', index=False)
    except Error as e:
        print("Error while writing file", e)


def get_n_features(features):
    """
    Return top n features (in DESC) by TARGET column
    :return:
    """
    sorted_features = pd.Series(features[TARGET].sort_values(ascending=[False]))
    top_n = pd.Series(sorted_features.head(TOP_N_FEATURES))
    result = pd.DataFrame(columns=['lookup', 'target'], data=features.loc[top_n.keys().values])
    print(result)
    return result


# End of Utils
