# automatic-feature-engineering-gkg
Automatic Feature Engineering using Freebase Graph

## Requirements
1. Python 3 or later
2. MySQL Database
3. Configuration in config.py
4. Database details & credentials.
5. Freebase's dump file url: SOURCE_PATH
6. Fill ALLOWED_ENTITIES with allowed categories.

All categories are placed in db/tables.py

## Run & Test
For Init DB, create tables, read Freebase file and parsing into DB. run:
 
$ python Parser.py

For build and test features from three datasets, run:

$ python Main.py build/test train_file_1 train_file_1 test_file train_table_1 train_file_2 test_table

For example:

$ python Main.py test movies_after_matching.csv books_after_matching.csv dataset_189_baseball_after_matching.csv film book baseball  

The table_name is for the DB table according to entity category.

You can replace and change the order of the training datasets and the testing dataset, for example:

book + movie -> baseball

book + baseball -> movie

baseball + movie -> book

