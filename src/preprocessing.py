import matplotlib as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format

def graph_targets(df):

    # TODO: fix this once datasets are added
    class_counts = df.groupby("Position").year.count()
    colors = sns.color_palette('pastel')[0:5]

    plt.title(label="Class Distributions")
    plt.pie(class_counts, labels = class_counts.index, colors = colors, autopct=autopct_format(class_counts))
    plt.show()


def balance_dataset(X, y, method="SMOTE"):
    techniques = {
        "SMOTE" : SMOTE(random_state=0),
        "ADASYN" : ADASYN(random_state=0),
        "RANDOM" : RandomOverSampler(random_state=0),
    }

    sampler = techniques[method]

    X, y = sampler.fit_resample(X, y,)

    return X, y

    