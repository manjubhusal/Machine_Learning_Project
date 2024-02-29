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


def calc_info_gain(impurity, y, selected_feature, threshold):
    # Create children & calculate their weighted avg. impurity
    left_idxs, right_idxs = split(selected_feature, threshold)
    n_left, n_right = len(left_idxs), len(right_idxs)
    n = len(y)

    if n_left == 0 or n_right == 0:
        return 0

    if impurity == 'entropy':
        parent_impurity = calc_entropy(y)
        e_left = calc_entropy(y[left_idxs])
        e_right = calc_entropy(y[right_idxs])
        child_impurity = (n_left / n) * e_left + (n_right / n) * e_right
    elif impurity == 'gini':
        parent_impurity = calc_gini(y)
        g_left, g_right = calc_gini(y[left_idxs]), calc_gini(y[right_idxs])
        child_impurity = (n_left / n) * g_left + (n_right / n) * g_right
    elif impurity == 'mis_error':
        parent_impurity = calc_misclass_error(y)
        m_left, m_right = (calc_misclass_error(y[left_idxs]),
                           calc_misclass_error(y[right_idxs]))
        child_impurity = (n_left / n) * m_left + (n_right / n) * m_right
    else:
        print("INFO GAIN CALC ERROR")

    information_gain = parent_impurity - child_impurity

    return information_gain


# This function splits the data into two groups (left and right) based on whether
# the data points fall below or above a given threshold value for a selected feature.
def split(X_column, split_thresh):
    left_idxs = np.where(X_column <= split_thresh)[0]
    right_idxs = np.where(X_column > split_thresh)[0]
    return left_idxs, right_idxs


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
