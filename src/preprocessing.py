import matplotlib as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

SEED = 42

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format

def scale_data(X_train, X_test, method="standard"):
    """
    Function to scale features....
    """
    techniques = {
        "standard" : StandardScaler(),
        "minmax" : MinMaxScaler(),
    }

    scaler = techniques[method]
    X_train_scaled = scaler.fit_transform(X_train)
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

def graph_targets(df):

    # TODO: fix this once datasets are added
    class_counts = df.groupby("Position").year.count()
    colors = sns.color_palette('pastel')[0:5]

    plt.title(label="Class Distributions")
    plt.pie(class_counts, labels = class_counts.index, colors = colors, autopct=autopct_format(class_counts))
    plt.show()






    