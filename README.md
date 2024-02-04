# P1_RandomForests

## Project Collaborators
Ester Aguilera, Jacob Graves and Manju Bhusal

Currently, this README is meant for developer use only.
***
## How to set up the project locally
We will be utilizing the following:
- Python version 3.12.1
- Pycharm 2022.2.5 (Community Edition)

Clone the project repo using the lobogit repo URL (ssh/https).
Remember to always pull before pushing anything to prevent version 
conflicts & please add a commit message to all commits.

## Tools we are allowed to use:
- Pandas library to deal with our data
- train_test_split or other data split methods from skleanr.model_selection
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

