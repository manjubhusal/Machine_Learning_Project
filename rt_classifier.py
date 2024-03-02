from statistics import mode
from sklearn.utils import resample
from dt_classifier import DecisionTree


class RandomForest:
    def __init__(self, ig_type, node_split_min, max_depth, num_features, num_trees):
        self.ig_type = ig_type
        self.node_split_min = node_split_min
        self.max_depth = max_depth
        self.num_features = num_features
        self.num_trees = num_trees
        pass

    def build_classifier(self, X_train, y_train, X_validate):
        all_predictions = []

        # 1. Bootstrapping, training & predicting
        for _ in range(self.num_trees):
            # 1. Create bootstrapping samples
            # If this method takes too long, we can try using the shuffle split
            # function to create bootstrap samples instead
            X_sample, y_sample = resample(X_train, y_train, replace=True,
                                          n_samples=len(X_train), random_state=42)
            # 2. Use samples to create n Decision Trees
            dt_classifier = DecisionTree(ig_type=self.ig_type, node_split_min=self.node_split_min,
                                         max_depth=self.max_depth, num_features=self.num_features)

            dt_classifier.fit(X_sample, y_sample)
            # We will use the full X_validate dataset for each tree
            predictions = dt_classifier.predict(X_validate)
            # Bagging our predictions
            all_predictions.append(predictions)

        # Holding Majority vote
        majority_prediction = mode(all_predictions)

        return majority_prediction

