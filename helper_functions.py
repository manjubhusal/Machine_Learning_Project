import pandas as pd
import numpy as np
from DTnode import DTnode


# "Metric functions"
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    ps = counts / total_samples
    gini = 1 * sum(ps ** 2)
    return gini


# entropy
# misclassification error

# Find the best split in each tree node
def find_best_split(X, y, metric_func):
    num_features = X.shape[1]
    best_feature = None
    best_threshold = None
    start_score = metric_func(y)
    best_info_gain = 0

    # Iterate over the different features
    for feature_index in range(num_features):
        # Find all possible splits
        thresholds = np.unique(X[:, feature_index])

        # Iterate over all possible splits for that given feature
        for threshold in thresholds:

            # Calculate IG
            left_indices = X[:, feature_index] <= threshold
            right_indices = left_indices
            if np.any(left_indices) and np.any(right_indices):
                score_left = metric_func(y[left_indices])
                score_right = metric_func(y[right_indices])
                score = ((np.sum(left_indices) / len(y)) * score_left +
                         (np.sum(right_indices) / len(y)) * score_right)
                info_gain = start_score - score

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold

    return best_feature, best_threshold


def build_tree_recursive(X, y, metric_func=gini):
    if len(np.unique(y)) == 1:
        # "pure" leaf, create a leaf node (majority vote)
        return DTnode(value=np.bincount(y).argmax())

    best_feature, best_threshold = find_best_split(X, y, metric_func)

    if best_feature is None:
        # No suitable split was found, create a lead node (majority vote)
        return DTnode(value=np.bincount(y).argmax())

    # recursive split
    left_indices = X[:, best_feature] <= best_threshold
    right_indices = left_indices

    left_subtree = build_tree_recursive(X[left_indices], y[left_indices], metric_func)
    right_subtree = build_tree_recursive(X[right_indices], y[right_indices], metric_func)

    return DTnode(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)


# Predict / Traverse the Tree
def predict_single(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature_index] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)


def predict(tree, X):
    y_hat = []
    for x in X:
        prediction = predict_single(tree, x)
        y_hat.append(prediction)
    y_hat = np.array(y_hat)
    return y_hat

# def calc_probs(label):
#     total_instances = len(label)
#     class_count = {}
#
#     for l in label:
#         if l in class_count:
#             class_count[l] += 1
#         else:
#             class_count[l] = 1
#
#     prob = [count / total_instances for count in class_count.values()]
#     return prob
#
#
# def impurity(prob):
#
#     e_prob = np.array(prob)
#     entropy = -np.sum(prob * np.log2(e_prob))
#
#     g_array = np.array(prob)
#     gini = -np.sum(g_array ** 2)
#
#     miss_array = np.array(prob)
#     miss_error = - np.max(miss_array)
#
#     final = max(entropy, gini, miss_error)
#
#     if (final == entropy):
#         print("Entropy is largest: ")
#
#     elif (final == gini):
#         print("Gini is largest: ")
#
#     else:
#         print("Miss_Error is largest: ")
#
#     return final
#
