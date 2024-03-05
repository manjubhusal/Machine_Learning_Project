from joblib import Parallel, delayed
from helper_functions import *


class DecisionTree:
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

    def __init__(self, X, y, ig_type, max_depth, node_split_min, num_features):
        self.ig_type = ig_type
        self.max_depth = max_depth
        self.node_split_min = node_split_min
        # make sure n_features is never set to be bigger than our actual number of features
        self.num_features = min(num_features, X.shape[1]) if num_features is not None else X.shape[1]
        self.current_depth = 0  # Initialize depth tracking variable
        self.count = 0  # TEST

        self.root = self.build_tree(X, y)

    @staticmethod
    def is_pure(labels):
        return len(np.unique(labels)) == 1

    def build_tree(self, X, y):
        # Increment depth at start of each call
        self.current_depth += 1

        # Create Node
        node = DecisionTree.Node()

        # Check stopping criteria
        if (self.is_pure(y) or
                self.max_depth <= self.current_depth or
                self.node_split_min > X.shape[0]):
            node.value = representative_class(y)
            # decrement depth before returning
            self.current_depth -= 1
            return node

        # Choose a randomized subset of features to consider for finding the best split
        subset = np.random.choice(X.shape[1], self.num_features, replace=False)

        # Find the optimal threshold for each feature in parallel
        results = Parallel(n_jobs=-1)(
            delayed(self.find_best_threshold)
            (feature_index, X[:, feature_index], X, y) for feature_index in subset
        )

        # Convert list of tuples to an array for easy access
        results_array = np.array(results)

        # Find the index of the maximum gain in the results array
        max_gain_index = np.argmax(results_array[:, 0])

        # Extract the split threshold, and feature index using the max_gain_index
        node.threshold = results_array[max_gain_index, 1]
        node.feature = int(results_array[max_gain_index, 2])  # Cast to int if necessary

        # Check if further splitting should occur based on the chi-square test
        if not should_split(X[:, node.feature], y):
            node.value = representative_class(y)
            self.current_depth -= 1
            return node

        # Split data into two groups (left and right) based on whether the data points
        # fall below or above a given threshold value for a selected feature.
        left = np.where(X[:, node.feature] <= node.threshold)[0]
        right = np.where(X[:, node.feature] > node.threshold)[0]

        # Use left and right index groups to create children
        node.left = self.build_tree(X[left, :], y[left])
        node.right = self.build_tree(X[right, :], y[right])

        self.current_depth -= 1
        return node

    def find_best_threshold(self, feature_index, selected_feature, X, y):
        thresholds = np.unique(selected_feature)
        gains_thresholds = [(calc_info_gain(self.ig_type, X, y, selected_feature, i), i) for i in thresholds]
        best_gain, best_threshold = max(gains_thresholds)
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
