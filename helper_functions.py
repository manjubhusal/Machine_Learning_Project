from collections import Counter
from scipy.stats import chi2
import numpy as np


def calc_gini(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    gini = 1 - sum(np.square(ps))
    return gini


def calc_misclass_error(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    miss_error = 1 - np.max(ps)
    return miss_error


def calc_entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum(ps * np.log(ps + 1e-12))


def chi_square(obs):
    total = np.sum(obs)
    expec = np.full_like(obs, total / len(obs))
    stat = np.sum((obs - expec) ** 2 / expec)
    df = len(obs) - 1
    p_val = 1 - chi2.cdf(stat, df)
    return stat, p_val


def should_split(attribute_values, class_labels):
    # Perform chi-square test
    attribute_values = np.array(attribute_values)
    class_labels = np.array(class_labels)
    observed_freq = np.histogram2d(attribute_values, class_labels)[0]
    chi_2, p = chi_square(observed_freq)

    # Check if any attribute is significantly correlated with the class
    if p.any() > 0.05:  # Adjust significance level as needed
        return False  # Stop splitting
    else:
        return True  # Continue splitting


# This function splits the data into two groups (left and right) based on whether
# the data points fall below or above a given threshold value for a selected feature.
def partition_data_by_threshold(X_column, split_thresh):
    left_idxs = np.where(X_column <= split_thresh)[0]
    right_idxs = np.where(X_column > split_thresh)[0]
    return left_idxs, right_idxs


# The old _most_common_label function is lines 49-51 after function name was changed
# def representative_class(y):
#     unique, counts = np.unique(y, return_counts=True)
#     return unique[np.argmax(counts)]

def representative_class(y):
    # In the case of our dataset, the majority class is '0' so that is what
    # we are returning
    if len(y) == 0:
        return 0
    counter = Counter(y)
    most_common_element = counter.most_common(1)
    if most_common_element:  # Check if the list is not empty
        return most_common_element[0][0]
    else:
        return 0
