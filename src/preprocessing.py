from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

SEED = 42

def scale_data(X_train=None, X_test=None, method="standard"):
    """
    Function to scale features....
    """
    techniques = {
        "standard" : StandardScaler(),
        "minmax" : MinMaxScaler(),
    }

    X_train_scaled, X_test_scaled = None, None

    scaler = techniques[method]

    if X_train is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def balance_dataset(X, y, method="SMOTE"):
    techniques = {
        "smote" : SMOTE(random_state=SEED),
        "random" : RandomOverSampler(random_state=SEED),
        "under" : RandomUnderSampler(random_state=SEED),
    }

    sampler = techniques[method]
    X_resampled, y_resampled = sampler.fit_resample(X, y,)

    return X_resampled, y_resampled

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format

def graph_distributions(y):

    class_counts = dict(Counter(y))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    colors = sns.color_palette('pastel')[0:5]

    plt.title(label="Class Distributions")
    plt.pie(counts, labels=classes, colors = colors, autopct=autopct_format(counts))
    plt.show()

def plot_classes(X, y):
    _, ax = plt.subplots(figsize=(6, 6))
    _ = ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolor="k")

def select_features(X, y, model=None, n_features="auto"):
    """
    n_features: int or float, default="auto"
        int: number of features to select
        float: percentage of features to select
        auto: select the number of features that gives the highest score based mean accuracy
    """

    selector = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward')
    selector.fit(X, y)

    return selector.get_support(indices=True)



