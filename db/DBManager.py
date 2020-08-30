import json

import mysql.connector
from mysql.connector import Error

from config_dev import database, host, password, port, username
from db.tables import TABLES


class DBManager:
    """
    """

    def __init__(self):
        """Initializing
        """
        self.db_connection = None

    def db_connect(self):
        """
        :return:
        """
        try:
            self.db_connection = mysql.connector.connect(host=host,
                                                         port=port,
                                                         user=username,
                                                         database=database,
                                                         password=password)
            if self.db_connection.is_connected():
                db_Info = self.db_connection.get_server_info()
                print("____________________________________________")
                print("Connected to MySQL Server version ", db_Info)
                cursor = self.db_connection.cursor()
                cursor.execute("select database();")
                record = cursor.fetchone()
                print("You're connected to database: ", record)
                print("____________________________________________")
                print()
            else:
                print("Not Connected!")

        except Error as e:
            print("Error while connecting to MySQL", e)

    def db_close(self):
        """
        :return:
        """
        if self.db_connection is not None and self.db_connection.is_connected():
            cursor = self.db_connection.cursor()
            cursor.close()
            self.db_connection.close()
            print("MySQL connection is closed")

    def get_db_connection(self):
        """
        :return:
        """
        return self.db_connection

    def is_connected(self):
        """
        :return:
        """
        if self.db_connection is None:
            return False
        return self.db_connection.is_connected()

    def db_execute_query(self, query):
        """
        :param query:
        :return:
        """
        if self.db_connection.is_connected():
            cursor = self.db_connection.cursor()
            result = cursor.execute(query)
            print("Query executed successfully")
            # Make sure data is committed to the database
            self.db_connection.commit()
            cursor.close()
            return result

    def db_create_database(self):
        """
        :return:
        """
        mySql_create_db_query = "CREATE DATABASE IF NOT EXISTS `freebase` CHARACTER SET utf8 COLLATE utf8_general_ci; "
        self.db_execute_query(mySql_create_db_query)

    def db_init(self):
        """
        Init Database with all tables
        :return:
        """
        query = """CREATE TABLE IF NOT EXISTS db_   schema ( 
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name TEXT NOT NULL, 
                        subject TEXT NOT NULL, 
                        data LONGTEXT NOT NULL
                    ) """
        self.db_execute_query(query)

        for table in TABLES:
            query = "CREATE TABLE IF NOT EXISTS " + table + " LIKE db_schema"
            self.db_execute_query(query)

    def db_insert(self, table, name, subject, data):
        """
        :param table:
        :param name:
        :param subject:
        :param data:
        :return:
        """
        if self.db_connection is not None and self.db_connection.is_connected():
            # and self.db_fetch_by_subject(table, subject, False) is None:
            query = "INSERT INTO " + table + " (`name`, `subject`, `data`) VALUES (%s, %s, %s)"
            cursor = self.db_connection.cursor()
            cursor.execute(query, (name, subject, data))
            # Make sure data is committed to the database
            self.db_connection.commit()
            cursor.close()

    def db_fetch(self, table, name):
        """
        :param table:
        :param name:
        :return:
        """
        if self.db_connection is not None and self.db_connection.is_connected():
            query = "SELECT * FROM " + table + " WHERE `name` LIKE %s"
            cursor = self.db_connection.cursor()
            cursor.execute(query, (name,))
            result = cursor.fetchall()

            fields = map(lambda x: x[0], cursor.description)
            result = [dict(zip(fields, row)) for row in result]

            # prepare entire data (str to json)
            for index, item in enumerate(result):
                if 'data' in item.keys() and len(item['data']) > 0:
                    result[index]['data'] = json.loads(result[index]['data'])

            cursor.close()

            return result

    def db_fetch_by_subject(self, table, subject, assign=True):
        """
        :param assign:
        :param table:
        :param subject:
        :return:
        """
        result = None
        if self.db_connection is not None and self.db_connection.is_connected():
            query = "SELECT * FROM " + table + " WHERE `subject` LIKE %s AND `name` <> ''"
            cursor = self.db_connection.cursor(buffered=True)
            cursor.execute(query, (subject,))
            result = cursor.fetchone()

            if assign and result and result is not None:
                # Assign names to values
                fields = map(lambda x: x[0], cursor.description)
                result = dict(zip(fields, result))

            cursor.close()

        return result
