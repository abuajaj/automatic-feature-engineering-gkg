from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from scipy import interp
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder

import Utils
from config_dev import LOOKUP, TARGET

plt.style.use('seaborn-whitegrid')


class Evaluator:
    """
    """

    def __init__(self) -> object:
        """Initializing
        """
        self.scoring_model = svm.SVC(kernel='linear', probability=True)
        self.model = linear_model.LinearRegression()

    def get_model(self):
        """
        """
        return self.model

    def evaluate(self, extracted_features, meta_features, dataset, classes):
        """
        :param extracted_features: extracted features dataset
        :param meta_features: meta features dataset
        :param dataset: original dataset
        :param classes:
        :return:
        """

        le = LabelEncoder()
        y = le.fit_transform(dataset[TARGET])
        classes = le.fit_transform(classes)
        dataset.drop([LOOKUP, TARGET], axis=1, inplace=True)

        # normalize
        for feature, feature_data in dataset.items():
            dataset[feature] = Utils.normalize(feature_data)

        meta_features[TARGET] = pd.Series()
        meta_features[TARGET].fillna(0, inplace=True)

        if not len(dataset.keys()):
            OLD_AUC = 0
        else:
            OLD_AUC = self.find_target(dataset, y, classes)

        for index, row in meta_features.iterrows():
            X = pd.concat([dataset, extracted_features[row[LOOKUP]]], axis=1, sort=False)
            NEW_AUC = self.find_target(X, y, classes)
            if OLD_AUC == 0:
                meta_features.set_value(index, TARGET, NEW_AUC)
            else:
                meta_features.set_value(index, TARGET, NEW_AUC - OLD_AUC)

        return meta_features

    def find_target(self, X, y, classes):
        """
        :param X:
        :param y:
        :param classes:
        :return:
        """
        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

        self.scoring_model.fit(X_train, y_train)

        y_pred = self.scoring_model.predict(X_test)

        try:
            AUC = roc_auc_score(label_binarize(y_test, classes=classes), label_binarize(y_pred, classes=classes))
            print(AUC)
            return AUC
        except ValueError:
            return 0

    def predict(self, X):
        """
        :param X:
        :return:
        """
        if np.any(np.isinf(X)):
            X.replace(np.inf, 0, inplace=True)
        if np.any(np.isnan(X)):
            X.replace(np.nan, 0, inplace=True)

        y_pred = self.model.predict(X)

        return y_pred

    def model(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))

        y = label_binarize(y, classes=list(set(y)))
        n_classes = y.shape[1]

        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

        print("Score:", classifier.score(X_test, y_test))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

        return roc_auc

    def model_2(self, X, y):
        """
        """
        # create training and testing vars
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # fit a model
        lm = linear_model.LinearRegression()
        model = lm.fit(X_train, y_train)
        predictions = lm.predict(X_test)

        # The line / model
        plt.scatter(y_test, predictions)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.plot()

        print("Score:", model.score(X_test, y_test))

    def model_3(self, X, y, input):
        # sns.pairplot(X)
        # sns.heatmap(X.corr(), annot=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        classifier = Sequential()
        # First Hidden Layer
        classifier.add(Dense(input, activation='relu', kernel_initializer='random_normal', input_dim=input * 2))
        # Second  Hidden Layer
        classifier.add(Dense(input, activation='relu', kernel_initializer='random_normal'))
        # Output Layer
        classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

        # Compiling the neural network
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fitting the data to the training dataset
        classifier.fit(X_train, y_train, batch_size=10, epochs=100)

        eval_model = classifier.evaluate(X_train, y_train)
        print(eval_model)

        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
