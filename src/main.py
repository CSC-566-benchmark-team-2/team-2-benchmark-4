from sklearn.base import RegressorMixin
from . import svm

def create_model() -> RegressorMixin:
    """
    Put any initialization logic for your model in this function.
    No need to `fit` or `predict`
    Initialize with all hyperparameters here
    Returns the initialized model
    """
    return svm.SVMClassifier(C=1, kernel='rbf', degree=3)
    pass
