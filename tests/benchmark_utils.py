import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error
from sklearn import datasets

def preprocess_video_games_df(path_to_video_games_csv):
    """
    This function will preprocesses the video games DF and returns the entire
    X and y dataframes.

    Input:
      path_to_video_games_csv (string): Path to csv
    Output:
      (X, y): Tuple of dataframes
        -> X : Dataframe with 5 columns of String Type
        -> y : Series with 1 column of float64 type
    """

    df = pd.read_csv(path_to_video_games_csv)
    df = df.drop(
        [
            "NA_Sales",
            "EU_Sales",
            "JP_Sales",
            "Other_Sales",
            "Critic_Score",
            "Critic_Count",
            "User_Score",
            "User_Count",
            "Developer",
            "Rating",
        ],
        axis=1,
    )

    # Set the year to categorical
    df["Year_of_Release"] = df["Year_of_Release"].apply(str)

    # Get rid of na rows
    df.dropna(inplace=True)

    # Set target variable
    y = df.pop("Global_Sales")
    return df, y


def preprocess_life_df(path_to_life_csv):
    """
    This function will preprocesses the life expectancy DF and returns the
    X and y dataframes.

    Input:
    path_to_life_csv (string): Path to csv
    Output:
    (X, y): Tuple of dataframes
        -> X : Dataframe with 7 columns of String Type
        -> y : Series with 1 column of float64 type
    """
    df = pd.read_csv(path_to_life_csv)

    # get y attrib
    y = df["Life expectancy "]

    # Type fixing
    df["Country"] = df["Country"].astype(str)
    df["Year"] = df["Year"].apply(str)
    df["Status"] = df["Status"].astype(str)

    # convert numerics to ranges
    per1000_bins = [i for i in range(0, 1001, 100)]
    per1000_labels = ["({i}-{j}]".format(i=i, j=i + 100) for i in per1000_bins[:-1]]
    per100_bins = [i for i in range(0, 101, 10)]
    per100_labels = ["({i}-{j}]".format(i=i, j=i + 10) for i in per100_bins[:-1]]
    per1_bins = [round(x * 0.1, 1) for x in range(0, 11)]
    per1_labels = ["({i}-{j}]".format(i=i, j=round(i + 0.1, 1)) for i in per1_bins[:-1]]

    df["Adult Mortality"] = pd.cut(
        df["Adult Mortality"], bins=per1000_bins, labels=per1000_labels
    ).astype(str)
    df["Hepatitis B %immun"] = pd.cut(
        df["Hepatitis B"], bins=per100_bins, labels=per100_labels
    ).astype(str)
    df["BMI"] = pd.cut(df[" BMI "], bins=per100_bins, labels=per100_labels).astype(str)
    df["Polio %immun"] = pd.cut(
        df["Polio"], bins=per100_bins, labels=per100_labels
    ).astype(str)
    df["Diphtheria %immun"] = pd.cut(
        df["Diphtheria "], bins=per100_bins, labels=per100_labels
    ).astype(str)
    df["Income composition of resources"] = pd.cut(
        df["Income composition of resources"], bins=per1_bins, labels=per1_labels
    ).astype(str)

    # selected features
    features = [
        "Country",
        "Year",
        "Status",
        "Adult Mortality",
        "Hepatitis B %immun",
        "BMI",
        "Polio %immun",
        "Diphtheria %immun",
        "Income composition of resources",
        "Life expectancy ",
    ]

    # only keeping main features
    df = df.loc[:, features]

    # drop missing values
    df = df.dropna()
    y_feature = "Life expectancy "
    y = df[y_feature]
    df = df.drop(y_feature, axis=1)
    return df, y


def preprocess_flare_df(**kwargs):
    df = pd.read_csv(kwargs["link"], sep=kwargs["sep"], names=kwargs["names"])

    # drop this because there is no variance
    df = df.drop(columns=["largest-spotarea"])

    # drop missing values
    df = df.dropna()

    # get y attrib
    y = df.pop("C")

    return df, y


def preprocess_titanic_df(path_to_titanic_csv):

    titanic_df = pd.read_csv(path_to_titanic_csv)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
    titanic_df2 = titanic_df.loc[:, features]
    titanic_df2["CabinLetter"] = titanic_df2["Cabin"].str.slice(0, 1)
    X = titanic_df2.drop("Cabin", axis=1)
    X["CabinLetter"] = X["CabinLetter"].fillna("?")
    X["Pclass"] = X["Pclass"].astype(str)
    X["SibSp"] = X["SibSp"].astype(str)
    X["Parch"] = X["Parch"].astype(str)
    X["Age"] = (
        ((X["Age"].fillna(X["Age"].mean()) / 10).astype(int) * 10)
        .astype(int)
        .astype(str)
    )

    X = X.dropna()

    X2 = X.drop(columns="Fare")
    t = X["Fare"]

    return X2, t

# y has 2 classes with 850 1s and 8423 -1s, unbalanced
def preprocess_pulsar_df(path_to_pulsar_csv):
    pulsar_df = pd.read_csv(path_to_pulsar_csv)
    pulsar_df.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness', 
              'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']
    
    X = pulsar_df.dropna()
    y_feature = "target_class"
    y = X[y_feature]
    X = X.drop(y_feature, axis=1)
    y2 = (y-0.5)*2
    return X.to_numpy(), y2.to_numpy()


def preprocess_heart_df(path_to_heart_csv):
    heart_df = pd.read_csv(path_to_heart_csv)
    
    X = heart_df.dropna()
    y_feature = "DEATH_EVENT"
    y = X[y_feature]
    X = X.drop(y_feature, axis=1)
    y2 = (y-0.5)*2
    return X.to_numpy(), y2.to_numpy()
    
# def linear_separable_1():
#     np.random.seed(1)
#     X, y = datasets.make_blobs(n_samples=300, centers=3, n_features=2, center_box=(0, 7))
#     y2 = (y-0.5)*2
#     return X, y2

# def linear_non_separable_1():
#     np.random.seed(1)
#     X, y = datasets.make_blobs(n_samples=300, centers=3, n_features=2, center_box=(0, 5))
#     y2 = (y-0.5)*2
#     return X, y2

# def non_linear_1():
#     X, y = datasets.make_circles(n_samples=300, noise=0.05)
#     y2 = (y-0.5)*2
#     return X, y2



def gaussian_quantiles():
    np.random.seed(1)
    X, y = datasets.make_gaussian_quantiles(n_classes=2, n_samples=1000)
    y2 = (y-0.5)*2
    return X, y2

def moons():
    np.random.seed(1)
    X, y = datasets.make_moons(n_samples=1000, noise = 0.1)
    y2 = (y-0.5)*2
    return X, y2

def breast_cancer():
    np.random.seed(1)
    X, y = datasets.load_breast_cancer(return_X_y=True)
    y2 = (y-0.5)*2
    return X, y2

def challenge1():
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=25,
                                        n_informative=8,
                                        n_repeated=3,
                                        n_classes=2,
                                        weights=[.3,.7],
                                        random_state=1)
    y2 = (y-0.5)*2
    return X, y2

def challenge2():
    X,y = datasets.make_classification(
                    n_samples = 1000,
                    n_features = 12,
                    n_informative = 7,
                    n_redundant = 3,
                    n_repeated = 2,
                    n_classes = 2,
                    # Distribution of classes 20% Output1
                    # 20%> output 2, 30% output 3 and 4        
                    weights = [.1,.9],
                    random_state = 1)
    y2 = (y-0.5)*2
    return X, y2

# def extra_challenge():
#     np.random.seed(1)
#     X = np.random.random((5,1000))
#     y = 0.001*X[0] + 3.8*np.log(X[1]) - X[2]**3 - 0.4*np.abs(X[4]) + 1.11*np.exp(X[5]) - 0.00001*np.sin(X[8])
#     y = np.where(y > 0, 1, -1)
#     return X.T,y


# def extra_challenge():
#     np.random.seed(1)
#     X = np.random.random((5,1000))
#     y = X[0] * X[1] * X[2] * X[3] * X[4]
#     y = np.where(y > np.mean(y), 1, -1)
#     return X.T,y


def extra_challenge():
    np.random.seed(1)
    X = np.random.random((10,1000))
    y = 0.001*X[0] + 3.8*np.log(X[1]) - X[2]**3 - 0.4*np.abs(X[4]) + 1.11*np.exp(X[5]) - 0.00001*np.sin(X[8])
    y = np.where(y > 0, 1, -1)
    return X.T,y