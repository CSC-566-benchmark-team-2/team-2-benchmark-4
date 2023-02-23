from sklearn.base import RegressorMixin
from tests.benchmark_utils import (
    preprocess_pulsar_df,
    preprocess_heart_df,
    gaussian_quantiles,
    moons,
    breast_cancer,
    challenge1,
    challenge2,
    balance_dataset, 
    scale_data
)
import pandas as pd
from . import svm
from preprocessing import balance_dataset, scale_data, select_features

class Solution:
    def __init__(self):
        self.datasets = {
            "gaussian": gaussian_quantiles(),
            "moons": moons(),
            "breast_cancer" : breast_cancer(),
            "challenge1": challenge1(),
            "challenge2": challenge2(),
            "pulsar": preprocess_pulsar_df("data/pulsar_star.csv"),
            "heart": preprocess_heart_df("data/heart_failure.csv"),
        }
    
    def create_model(self) -> RegressorMixin:
        """
        Put any initialization logic for your model in this function.
        No need to `fit` or `predict`
        Initialize with all hyperparameters here
        Returns the initialized model
        """
        return svm.SVMClassifier(C=1, kernel='rbf', degree=3)
        pass

    def modify_datasets(self):
        self.datasets['challenge1'] = self._challenge1()
        self.datasets['challenge2'] = self._challenge2()
        self.datasets['pulsar'] = self._challenge3()
        self.datasets['heart'] = self._challenge4()

    def _challenge1(self):
        X1, y1 = self.datasets['challenge1']

        # add your code here!

        return X1, y1

    def _challenge2(self):
        X2, y2 = self.datasets['challenge2']

        # add your code here!

        return X2, y2
    
    def _challenge3(self):
        X3, y3 = self.datasets['pulsar']

        # add your code here!

        return X3, y3
        
    def _challenge4(self):
        X4, y4 = self.datasets['heart']

        # add your code here!

        return X4, y4
