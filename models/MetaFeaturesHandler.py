import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from config_dev import LOOKUP

plt.style.use('seaborn-whitegrid')


class MetaFeaturesHandler:
    """
    """

    def __init__(self) -> object:
        """Initializing
        """
        self.model = None
        self.meta_features = pd.DataFrame()

    def build(self, X, y):
        # Create a Gaussian Classifier
        self.model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                            max_depth=None, max_features='auto', max_leaf_nodes=None,
                                            min_impurity_decrease=0.0, min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=2,
                                            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                                            oob_score=False, random_state=None, verbose=0,
                                            warm_start=False)

        # Train the model using the training sets
        self.model.fit(X, y)
        # Find meta features
        self.features_importance(X)
        self.features_scores(X, y)

        return self.meta_features

    def features_importance(self, X):
        """
        Find features importance
        :param X:
        :return:
        """
        result = pd.Series(self.model.feature_importances_, index=X.keys(), name='importance') \
            .sort_values(ascending=False)
        print('feature_imp', result)

        # Creating a bar plot
        sns.barplot(x=result, y=result.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.show()

        self.meta_features[LOOKUP] = list(result.keys())
        self.meta_features['importance'] = list(result.values)

    def features_scores(self, X, y):
        """
        Calculate features scores: chi2, F - score, mutual info
        :param X:
        :param y:
        :return:
        """
        self.meta_features['chi2_score'], self.meta_features['chi_2_p_value'] = chi2(X, y)
        self.meta_features['f_score'], self.meta_features['f_p_value'] = f_classif(X, y)
        self.meta_features['mut_info_score'] = mutual_info_classif(X, y)

        print('chi2 score        ', self.meta_features['chi2_score'])
        print('chi2 p-value      ', self.meta_features['chi_2_p_value'])
        print('F - score score   ', self.meta_features['f_score'])
        print('F - score p-value ', self.meta_features['f_p_value'])
        print('mutual info       ', self.meta_features['mut_info_score'])

        print('chi2 score mean        ', np.mean(self.meta_features['chi2_score']))
        print('F - score score mean   ', np.mean(self.meta_features['f_score']))
        print('mutual info mean       ', np.mean(self.meta_features['mut_info_score']))
