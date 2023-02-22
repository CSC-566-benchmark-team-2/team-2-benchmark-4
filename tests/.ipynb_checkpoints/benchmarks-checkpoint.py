import sys
import os

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from benchmark_utils import (
    preprocess_flare_df,
    preprocess_life_df,
    preprocess_video_games_df,
    preprocess_pulsar_df,
    preprocess_heart_df,
    gaussian_quantiles,
    moons,
    breast_cancer,
    challenge1,
    challenge2,
    extra_challenge
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import timeit


def run_benchmarks(model_func, what_model_to_test = "svm") -> dict:
    SEED = 42
    TEST_SIZE = 0.2
    TIMING_ITERATIONS = 3
    
    
    bagging_datasets = {
        "video_games": preprocess_video_games_df("data/video_games.csv"),
        "life": preprocess_life_df("data/life_expectancy.csv"),

    }
    
    svm_datasets = {
        "gaussian": gaussian_quantiles(),
        "moons": moons(),
        "breast_cancer" : breast_cancer(),
        "challenge1": challenge1(),
        "challenge2": challenge2(),
        "extra_challenge": extra_challenge(),
        "pulsar": preprocess_pulsar_df("data/pulsar_star.csv"),
        "heart": preprocess_heart_df("data/heart_failure.csv"),
    }
    # for dataset_name, (X, y) in svm_datasets.items():
    #     print(dataset_name, X, y)
        
    np.random.seed(SEED)
    results = {}
    
    if what_model_to_test == "bagging":
        datasets = bagging_datasets
    elif what_model_to_test == "svm":
        datasets = svm_datasets
        
        
    for dataset_name, (X, y) in datasets.items():
        n_iterations = 1 if dataset_name == "video_games" else TIMING_ITERATIONS
        model = model_func()
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True
        )
        preds = None
        print("Fitting the dataset", dataset_name, "...")
        def benchmark():
            nonlocal preds
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            print("error rate is:", np.sum(np.sign(preds)!=np.sign(y_test))/len(y_test))

        perf = timeit.timeit(benchmark, number=n_iterations) / n_iterations
        
        if what_model_to_test == "bagging":
            results[dataset_name] = {
                "mse": mean_squared_error(y_test, preds),
                "runtime": perf,
            }
        elif what_model_to_test == "svm":
            results[dataset_name] = {
                "error rate": np.sum(np.sign(preds)!=np.sign(y_test))/len(y_test),
                "runtime": perf,
            }
    return results
