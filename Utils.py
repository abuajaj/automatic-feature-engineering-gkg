import json
import operator
import re
from datetime import datetime

import numpy as np

from config_dev import NOT_ALLOWED_TYPES


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
    return s.replace("@en", "").replace("\"", "").replace("\\", "")


def is_english(s):
    """
    :param s:
    :return:
    """
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def is_english2(s):
    """
    :param s:
    :return:
    """
    if (re.search("\"@", s) or re.search("@", s)) and not re.search("\"@en", s):
        return False
    else:
        return True


def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "Current Time =", current_time


def print_json(data):
    json_formatted_str = json.dumps(data, indent=2)
    print(json_formatted_str)
    print()


def missing_data(feature_type, values):
    """
    :return:
    """

    if feature_type == int or feature_type == float:
        if feature_type == int:
            values = [int(x) for x in values]
        if feature_type == float:
            values = [float(x) for x in values]

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
    column_types = dict()
    for value in column:
        value_type = type(value)
        if value_type is str:
            try:
                print(value)
                tmp = float(value)
                if isinstance(tmp, (int, float, complex)) and not isinstance(tmp, bool):
                    value_type = float
            except ValueError:
                pass
        if value_type in column_types:
            column_types[value_type] = column_types[value_type] + 1
        else:
            column_types[value_type] = 1
    return max(column_types.items(), key=operator.itemgetter(1))[0]


def sort_keys(items):
    return {k: v for k, v in reversed(sorted(items.items(), key=lambda item: item[1]))}

# End of Utils
