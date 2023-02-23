# Team-2 Benchmark-3 Student Repo

## Repository Structure

- /data: folder containg datasets used for benchmark tasks
- /notebooks: notebooks to run and test code written in /src
- /src: source code that will be autograded in GitHub


## Idea Behind Benchmark 3
We have gather/created 5 datasets with various challenges encountered in real life data such as class imbalances, outliers, extraneous features etc. We have give you a few "tools" to deal with the aformentioned problems and your goal is to use these tools, with your SVM implementation, to try and get the best accuracy. Results will be displayed in the usual [web-app](https://csc-566-benchmark-results.netlify.app/) as last time. 

## Development Flow
- benchmark_3.ipynb: Prototype / Experiment
- main.py: Paste solutions from notebook into `create_model` and `_challenge1`, `_challenge2`... to submit 
- svm.py: The imports in main.py assume a python file svm.py in the same directory as main.py with code containing your svm implementation.

**NOTE:** Copies of the preprocessing function are imported into main.py, but if you change the functions in the notebook you must also change them in preprocessing.py

## Submission Process

1. In `src/main.py`, fill out the `create_model` function. This function should initialize and return the model that you want to use in the benchmarks.
2. Copy preprocessing code from 'Challenge _' sections in notebook to the Solution class in `src/main`.
3. Push your code to submit. The benchmarks will run in a GitHub action. You can submit as many times as you want. Your latest submission will appear on the leaderboard.
4. After a few minutes, check the [leaderboard site](https://csc-566-benchmark-results.netlify.app/) to see your results.

## Setup a virtual env

From the command line in the root directory run:

1. `python3.9 -m venv .venv`
2. `source .venv/bin/activate`
3. ` pip install -r requirements.txt`
