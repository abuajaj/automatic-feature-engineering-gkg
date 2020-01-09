from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


class Selection:
    """
    """

    def __init__(self) -> object:
        """Initializing
        """
        self.model = None

    def get_model(self):
        """
        """
        return self.model

    def build_model(self):
        """
        """

    # load the dataset
    def prepare_dataset(self, data):
        # split into input (X) and output (y) variables
        X = data.loc[:, ~data.columns.isin(['Player', 'Hall_of_Fame'])].values
        y = data['Hall_of_Fame']
        # format all fields as string
        X = X.astype(str)
        return X, y

    # prepare input data
    def prepare_inputs(self, X_train, X_test):
        print(X_train)
        print(X_test)

        oe = OrdinalEncoder()
        oe.fit(X_train)
        X_train_enc = oe.transform(X_train)
        X_test_enc = oe.transform(X_test)
        return X_train_enc, X_test_enc

    # prepare target
    def prepare_targets(self, y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        return y_train_enc, y_test_enc

    # feature selection
    def select_features(self, X_train, y_train, X_test):
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs
