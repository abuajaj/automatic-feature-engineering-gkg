"""Python """
import gzip
import json
import re

from mysql.connector import Error

import Utils
from config_dev import ALLOWED_ENTITIES, SOURCE_PATH
from db.DBManager import DBManager


class Parser:
    """
    Parser Freebase triples into DB tables according to entity type
    """

    def __init__(self):
        """
        Initializing
        """
        self.db_manager = DBManager()

    def init_database(self):
        """
        :return:
        """
        if self.db_manager is not None and not self.db_manager.is_connected():
            self.db_manager.db_connect()
            self.db_manager.db_init()
            return self.db_manager.is_connected()
        return False

    def read_data(self, file):
        """
        :param file:
        :return:
        """
        try:
            iTotal = 0
            current_mid = ""
            current_topic = dict()
            with gzip.open(file, 'rt') as f:
                for line in f:
                    subject, predicate, object = Utils.parse_triple(line)
                    if subject == current_mid:
                        if predicate not in current_topic:
                            current_topic[predicate] = [object]
                        else:
                            current_topic[predicate].append(object)
                    elif current_mid:
                        self.prepare_to_save(subject, current_topic)

                        current_topic.clear()

                    current_mid = subject

                    iTotal = iTotal + 1
                    if 0 == (iTotal % 1000000):
                        print("iTotal: ", iTotal)
                        print()

        except Error as e:
            print("Error while reading file", e)

    def prepare_to_save(self, subject, current_topic):
        """
        :param subject:
        :param current_topic:
        :return:
        """
        if '/type/object/type' in current_topic:
            for iType in current_topic['/type/object/type']:
                for allowed_type_key, allowed_type_table in ALLOWED_ENTITIES.items():
                    if re.search(allowed_type_key, iType):
                        # Save to DB
                        if self.prepare_database():
                            self.save_to_database(allowed_type_table, subject, current_topic)
                            break

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

    def save_to_database(self, table, subject, current_topic):
        """
        :param table:
        :param subject:
        :param current_topic:
        :return:
        """
        current_topic = Utils.handle_duplicate(current_topic)
        current_topic = Utils.handle_language(current_topic)

        if current_topic is None or len(current_topic) == 0:
            return

        name = ''
        if '/type/object/name' in current_topic.keys():
            if isinstance(current_topic['/type/object/name'], list):
                name = current_topic['/type/object/name'][0]
            else:
                name = current_topic['/type/object/name']
            current_topic.pop('/type/object/name')
        elif 'label' in current_topic.keys():
            if isinstance(current_topic['label'], list):
                name = current_topic['label'][0]
            else:
                name = current_topic['label']
            current_topic.pop('label')

        if name and current_topic is not None and len(current_topic) > 0:
            name = Utils.clean_lang(name)
            self.db_manager.db_insert(table, name, subject, json.dumps(current_topic))


if __name__ == '__main__':
    print('>>> Start.. ', Utils.print_time())

    # init
    _parser = Parser()

    # init & prepare DB: create new entities tables
    _parser.init_database()

    # reading & parsing into DB
    _parser.read_data(SOURCE_PATH)
    _parser.db_manager.db_close()

    print('parser: The End is near!')
