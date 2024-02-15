import numpy as np
from collections import Counter
from scipy.stats import chi2_contingency
from scipy.stats import chisquare


def should_split(attribute_values, class_labels):
    # Remove non-finite values
    attribute_values = attribute_values[np.isfinite(attribute_values)]
    class_labels = class_labels[np.isfinite(class_labels)]

    # Perform chi-square test
    observed_freq = np.histogram2d(attribute_values, class_labels)[0]
    chi2, p = chisquare(observed_freq)

    # Check if any attribute is significantly correlated with the class
    if p.any() > 0.05:  # Adjust significance level as needed
        return False  # Stop splitting
    else:
        return True  # Continue splitting


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        # We need to make sure "n_features" does not exceed the number of actual features we have
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))


        # 1. check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or
                n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # This allows us to have a randomized group that does not contain duplicate features
        # feat_idxs is feature indices
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # 2. find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # 2.1 Check if further splitting should occur based on the chi-square test
        attribute_values = X[:, best_feature]
        if not should_split(attribute_values, y):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 3. create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    # Here we want to find all possible thresholds and splits and what the best ones are
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Calculate information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # 1. Get parent entropy
        parent_entropy = self._entropy(y)

        # 2. create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # 3. calculate the weighted avg. entropy of children
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # 4. calculate the IG
        information_gain = parent_entropy - child_entropy

        return information_gain

    # Calculates what the left & right indices should be
    def _split(self, X_column, split_thresh):
        # which indices will go to the left & which will go to the right
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        # creates sort of like a histogram for our array / tells us how many times
        # each value has occurred
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
