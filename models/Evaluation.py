from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import Sequential
from keras.layers import Dense
from scipy import interp
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

plt.style.use('seaborn-whitegrid')


class Evaluation:
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

    def features_importance(self, X, y):
        result = dict()
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 80% training and 20% test
        # Create a Gaussian Classifier
        clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                     max_depth=None, max_features='auto', max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, min_impurity_split=None,
                                     min_samples_leaf=1, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                                     oob_score=False, random_state=None, verbose=0,
                                     warm_start=False)
        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)

        result['feature_imp'] = pd.Series(clf.feature_importances_, index=X.keys()).sort_values(ascending=False)
        print('feature_imp', result['feature_imp'])

        # Creating a bar plot
        sns.barplot(x=result['feature_imp'], y=result['feature_imp'].index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.show()

        result['chi2_score'], result['chi_2_p_value'] = chi2(X, y)
        result['f_score'], result['f_p_value'] = f_classif(X, y)
        result['mut_info_score'] = mutual_info_classif(X, y)

        print('chi2 score        ', result['chi2_score'])
        print('chi2 p-value      ', result['chi_2_p_value'])
        print('F - score score   ', result['f_score'])
        print('F - score p-value ', result['f_p_value'])
        print('mutual info       ', result['mut_info_score'])

        print('chi2 score mean        ', np.mean(result['chi2_score']))
        print('F - score score mean   ', np.mean(result['f_score']))
        print('mutual info mean      ', np.mean(result['mut_info_score']))

        return result

    def model_1(self, X, y):
        """
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

    def evaluate(self):
        pass
