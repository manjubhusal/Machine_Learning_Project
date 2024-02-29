from joblib import Parallel, delayed
from helper_functions import *


class DecisionTree:
    def __init__(self, ig_type, max_depth, node_split_min, num_features):
        self.root = None
        self.ig_type = ig_type
        self.max_depth = max_depth
        self.node_split_min = node_split_min
        self.num_features = num_features
        self.count = 0  # TEST

    def fit(self, X, y):
        # We need to make sure "n_features" does not exceed the number of actual
        # features we have
        if self.num_features is None:
            self.num_features = X.shape[1]
        else:
            if self.num_features > X.shape[1]:
                self.num_features = X.shape[1]
        # Infer the number of classes from the target array
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        # Create Node & label the node w most representative class
        node = Node()
        num_samples, num_features = X.shape

        # Check stopping criteria
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or
                num_samples < self.node_split_min):
            node.value = representative_class(y)
            return node

        # Choose a randomized subset of features to consider for finding the best split
        # feature_indices = np.random.choice(num_features, self.num_features, replace=False)
        feature_indices = np.random.permutation(num_features)[:self.num_features]
        best_threshold, best_feature_index = self.find_best_split(X, y, feature_indices)

        # Check if further splitting should occur based on the chi-square test
        if not should_split(X[:, best_feature_index], y):
            node.value = representative_class(y)
            return node

        # Create child nodes
        left_indices, right_indices = split(X[:, best_feature_index], best_threshold)
        node.feature = best_feature_index
        node.threshold = best_threshold
        node.left = self.build_tree(X[left_indices, :], y[left_indices], depth + 1)
        node.right = self.build_tree(X[right_indices, :], y[right_indices], depth + 1)

        return node

    # Here we want to find all possible thresholds and splits and what the best ones are
    # This function was previously called _best_split
    def find_best_split(self, X, y, feature_indices):
        # Find the optimal threshold for each feature in parallel
        results = Parallel(n_jobs=-1)(
            delayed(self.find_best_threshold)
            (feature_index, X[:, feature_index], y) for feature_index in feature_indices
        )

        # Convert list of tuples to an array for easy access
        results_array = np.array(results)

        # Find the index of the maximum gain in the results array
        max_gain_index = np.argmax(results_array[:, 0])

        # Extract the split threshold, and feature index using the max_gain_index
        # best_gain = results_array[max_gain_index, 0]
        split_threshold = results_array[max_gain_index, 1]
        split_feature_index = int(results_array[max_gain_index, 2])  # Cast to int if necessary

        return split_threshold, split_feature_index

    def find_best_threshold(self, feature_index, selected_feature, y):
        thresholds = np.unique(selected_feature)
        best_gain = -1
        best_threshold = None

        for threshold in thresholds:
            calculated_gain = calc_info_gain(self.ig_type, y, selected_feature, threshold)
            if calculated_gain > best_gain:
                best_gain, best_threshold = calculated_gain, threshold

        return best_gain, best_threshold, feature_index

    def predict(self, X):
        # Initialize an empty list to store the predictions
        predictions = []

        # Loop through each item in X
        for x in X:
            # Traverse the tree starting from the root for each item
            # and append the result to the predictions list
            # prediction = depth_first_traversal(x, self.root)
            prediction = self.classify(x, self.root)
            predictions.append(prediction)

        # Convert the list of predictions to a NumPy array before returning
        return np.array(predictions)

    def classify(self, x, node):
        # Base case: if the current node is a leaf node
        if node.is_leaf():
            return node.value
        # Recursive case: traverse the tree based on the feature value of x
        else:
            if x[node.feature] <= node.threshold:
                return self.classify(x, node.left)
            else:
                return self.classify(x, node.right)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.value = value
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold

    def is_leaf(self):
        if self.value is not None:
            return True
        else:
            return False
