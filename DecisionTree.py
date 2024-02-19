import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2


# Functions that have been optimized:
# - Entropy, Gini, Mis. Class Error
# - Many attempts have been made to optimize best_split
#   with some improvement but is still the most computationally
#   exhaustive function we have

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
    chi2, p = chi_square(observed_freq)

    # Check if any attribute is significantly correlated with the class
    if p.any() > 0.05:  # Adjust significance level as needed
        return False  # Stop splitting
    else:
        return True  # Continue splitting


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None, ig_type=''):
        self.num_classes = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.ig_type = ig_type
        self.count = 0  # TEST

    def fit(self, X, y):
        # We need to make sure "n_features" does not exceed the number of actual features we have
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        # Infer the number of classes from the target array
        self.num_classes = len(np.unique(y))
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
        # growTree_start = time.perf_counter()  # BENCHMARK TEST
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)
        # growTree_end = time.perf_counter()  # BENCHMARK TEST
        # growTree_elapsed = growTree_end - growTree_start  # BENCHMARK TEST
        # print("Finding the ", self.count, "th best split took (in secs): ", growTree_elapsed)  # BENCHMARK TEST
        # self.count += 1  # BENCHMARK TEST

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

        def calculate_gain(feat_idx, X_column, y):
            thresholds = np.unique(X_column)
            local_best_gain = -1
            local_split_threshold = None
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > local_best_gain:
                    local_best_gain = gain
                    local_split_threshold = thr
            return local_best_gain, local_split_threshold, feat_idx

        results = Parallel(n_jobs=-1)(
            delayed(calculate_gain)(feat_idx, X[:, feat_idx], y) for feat_idx in feat_idxs
        )

        for gain, thr, feat_idx in results:
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

        if self.ig_type == 'entropy':
            e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
            child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        elif self.ig_type == 'gini':
            g_left, g_right = self._gini(y[left_idxs]), self._gini(y[right_idxs])
            child_entropy = (n_left / n) * g_left + (n_right / n) * g_right
        elif self.ig_type == 'mis_error':
            m_left, m_right = self._miss_error(y[left_idxs]), self._miss_error(y[right_idxs])
            child_entropy = (n_left / n) * m_left + (n_right / n) * m_right
        else:
            print("INFO GAIN CALC ERROR")

        # 4. calculate the IG
        information_gain = parent_entropy - child_entropy

        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.where(X_column <= split_thresh)[0]
        right_idxs = np.where(X_column > split_thresh)[0]
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y, minlength=self.num_classes)
        ps = hist / len(y)
        return -np.sum(ps * np.log(ps + 1e-12))

    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        gini = 1 - sum(np.square(ps))
        return gini

    def _miss_error(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        miss_error = 1 - np.max(ps)
        return miss_error

    def _most_common_label(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

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
