
from joblib import Parallel, delayed

from helper_functions import *


# def chi_square(obs):
#     total = np.sum(obs)
#     expec = np.full_like(obs, total / len(obs))
#     stat = np.sum((obs - expec) ** 2 / expec)
#     df = len(obs) - 1
#     p_val = 1 - chi2.cdf(stat, df)
#     return stat, p_val
#
#
# def should_split(attribute_values, class_labels):
#     # Perform chi-square test
#     attribute_values = np.array(attribute_values)
#     class_labels = np.array(class_labels)
#     observed_freq = np.histogram2d(attribute_values, class_labels)[0]
#     chi_2, p = chi_square(observed_freq)
#
#     # Check if any attribute is significantly correlated with the class
#     if p.any() > 0.05:  # Adjust significance level as needed
#         return False  # Stop splitting
#     else:
#         return True  # Continue splitting


# def representative_class(y):
#     unique, counts = np.unique(y, return_counts=True)
#     return unique[np.argmax(counts)]

# def representative_class(y):
#     # In the case of our dataset, the majority class is '0' so that is what
#     # we are returning
#     if len(y) == 0:
#         return 0
#     counter = Counter(y)
#     most_common_element = counter.most_common(1)
#     if most_common_element:  # Check if the list is not empty
#         return most_common_element[0][0]
#     else:
#         return 0


def _split(X_column, split_thresh):
    left_idxs = np.where(X_column <= split_thresh)[0]
    right_idxs = np.where(X_column > split_thresh)[0]
    return left_idxs, right_idxs


class DecisionTree:
    def __init__(self, ig_type, max_depth, node_split_min, n_features, n_classes):
        self.root = None
        self.ig_type = ig_type
        self.max_depth = max_depth
        self.node_split_min = node_split_min
        self.n_features = n_features
        self.count = 0  # TEST
        self.n_classes = n_classes

    def fit(self, X, y):
        # We need to make sure "n_features" does not exceed the number of actual features we have
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        # Infer the number of classes from the target array
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # 1. check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or
                n_samples < self.node_split_min):
            leaf_value = representative_class(y)
            return Node(value=leaf_value)

        # This allows us to have a randomized group that does not contain duplicate features
        # feat_idxs is feature indices
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # 2. find the best split
        # growTree_start = time.perf_counter()  # BENCHMARK TEST
        best_feature, best_thresh = self.find_best_split(X, y, feat_idxs)
        # growTree_end = time.perf_counter()  # BENCHMARK TEST
        # growTree_elapsed = growTree_end - growTree_start  # BENCHMARK TEST
        # print("Finding the ", self.count, "th best split took (in secs): ", growTree_elapsed)  # BENCHMARK TEST
        # self.count += 1  # BENCHMARK TEST

        # 2.1 Check if further splitting should occur based on the chi-square test
        attribute_values = X[:, best_feature]
        if not should_split(attribute_values, y):
            leaf_value = representative_class(y)
            return Node(value=leaf_value)

        # 3. create child nodes
        left_idxs, right_idxs = _split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    # Here we want to find all possible thresholds and splits and what the best ones are
    # This function was previously called _best_split
    def find_best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        def calculate_gain(feat_idx, X_column, y):
            thresholds = np.unique(X_column)
            local_best_gain = -1
            local_split_threshold = None
            for thr in thresholds:
                gain = self.calc_info_gain(y, X_column, thr)
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

    # This function was previously called _information_gain
    def calc_info_gain(self, y, X_column, threshold):

        # Create children & calculate their weighted avg. impurity
        left_idxs, right_idxs = _split(X_column, threshold)
        n_left, n_right = len(left_idxs), len(right_idxs)
        n = len(y)

        if n_left == 0 or n_right == 0:
            return 0

        if self.ig_type == 'entropy':
            parent_impurity = calc_entropy(y)
            e_left = calc_entropy(y[left_idxs])
            e_right = calc_entropy(y[right_idxs])
            child_impurity = (n_left / n) * e_left + (n_right / n) * e_right
        elif self.ig_type == 'gini':
            parent_impurity = calc_gini(y)
            g_left, g_right = calc_gini(y[left_idxs]), calc_gini(y[right_idxs])
            child_impurity = (n_left / n) * g_left + (n_right / n) * g_right
        elif self.ig_type == 'mis_error':
            parent_impurity = calc_misclass_error(y)
            m_left, m_right = (calc_misclass_error(y[left_idxs]),
                               calc_misclass_error(y[right_idxs]))
            child_impurity = (n_left / n) * m_left + (n_right / n) * m_right
        else:
            print("INFO GAIN CALC ERROR")

        information_gain = parent_impurity - child_impurity

        return information_gain

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
