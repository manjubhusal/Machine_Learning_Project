# P1_RandomForests

## Project Collaborators
Ester Aguilera, Jacob Graves and Manju Bhusal

Currently, this README is meant for developer use only.
***
## How to set up the project locally
We will be utilizing the following:
- Python 3.12.1
- Pycharm 2022.2.5 (Community Edition)
- Pandas 2.2.0 (install/add on Pycharm)
- Numpy 1.26.3 (install/add on Pycharm)
- Scipy 1.12.0 (install/add on Pycharm)
- scikit-learn 1.4.0 (install/add on Pycharm)
- pip (if you don't already have it; steps below)
```commandline
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
pip --version
```
- Kaggle API 1.6.5 (installation below)
  - Go to https://www.kaggle.com/settings/account and "create a new API token".
  - This will download a kaggle.json file you will need. 
  - Commands below are for Mac/Linux but process should be similar for Windows.
  - Line 1 installs kaggle; if you get permission errors, try: ```pip install --user kaggle```.
  - Line 2 is to move the kaggle.json file you downloaded to ~/.kaggle/kaggle.json.
  - Line 3 is to ensure that other users of your computer do not have read access to your credentials.
  - Line 4 is to ensure your have the correct API version.
```commandline
pip install kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle --version
```

**To install or add packages on Pycharm**, go to Pycharm's
settings and type in "interpreter"; select "python interpreter"
which should show you a list of your currently added python packages.
Here you can install any packages your missing. Some packages may already 
installed and only need to be added.


**Clone the project repo** using the lobogit repo URL (ssh/https).
Remember to always pull before pushing anything to prevent version 
conflicts & please add a commit message to all commits.

In order to **download the project data** you must first install the Kaggle API.
After you've done all the steps, on the terminal, navigate to where you want to 
dump the data files and run the following command:
```commandline
kaggle competitions download -c cs429529-project1-random-forests
```

***If you have any issues setting up the environment, please DM 
me on discord. - Ester***
## Tools we are allowed to use:
- Pandas library to deal with our data
- train_test_split or other data split methods from skleanr.model_selection
  (**we will be using scikit-learn** instead of skleanr since the latter is a 
   deprecated library)
- Any visualization tool or analysis library to obtain insights from your 
results
- scipy.stats.chi2.ppf

## Project goal

Code the following from scratch:
- Decision tree that uses IG as split criterium and chi square as alternate 
termination rule.
- Implement Information Gain (IG) with: 1. Entropy, 2. Gini Index, 3. Misclassification Error
- Implement a random forest based on your decision tree and any criteria 
of your choosing.

Perform the following experiments:

1. Compare & contrast the trees built by IG with entropy, gini index & ME, 
in terms of their structural properties and accuracy. Include for example 
maximum depth, average depth, average accuracy, etc.
2. Use chi-square as a termination rule with alpha = 0.01, 0.05, 0.1, 0.25, 
0.5, 0.9. Compare & contrast the resulting trees.

***

## Major Changes in Version 2.0
1. Most function names have been changed
2. Functions that could be moved outside of Decision Tree file have begun to be moved into
a helper_functions file. 
3. Previously our program was using entropy to calculate the parent's impurity for both
gini and misclassification error which was very wrong of course so that was corrected, however,
it created major issues when running the program using mis_error. More details in #3.
4. **representative_class()/_most_common_label()** now handles cases where y is empty in which
case it will return the most common class. This issue only happened whenever we ran a corrected
version of our program using ig_type=mis_error. The issue never happened with gini index or
entropy.


## Program Flow

    # PROCESS DATA:
    # (1) Split data into: IDs, Cat. features, Num. features, Classes
    # (2) Imputate numerical & categorical features
    # (3) Encode categorical features
    # (4) Concatenate num + cat features back into one
    # (5) TODO: attach IDs back to corresponding rows of features/classes
    # (6) Split data into "train" and "test" 
    # MAKE DTs -> TRAIN -> PREDICT -> TEST ACCURACY
    # TESTING ACCURACY:
    # 1. Calculate the True Negative(TN), FalsePositive(FP), FalseNegative (FN)
    #    and TruePositive(TP) and put them in an array
    # 2. Calculate True Positive Rate(TPR) and True Negative Rate(TNR)
    # 3. Calculate Balanced Accuracy
    # Process is available to be implemented via entropy, mis. class error
    # and gini index.


    # Random Forest
    # TRAIN:
    # - Get a subset of the dataset
    # - Create a DT
    # - Repeat for as many times as the number of trees
    # TEST:
    # - Get the predictions from each tree
    # - Classification: hold a majority vote
    # - Regression: get the mean of the predictions