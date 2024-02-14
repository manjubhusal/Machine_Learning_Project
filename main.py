import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from helper_functions import build_tree_recursive
from helper_functions import predict

# # Read in the data
# df = pd.read_csv('data/train.csv')
#
# # Split the data into tID / features / target
# tID = df.iloc[:, 0].values  # don't know if this is really necessary?
# x = df.iloc[:, 1:25].values  # ALL features
# y = df.iloc[:, 26].values  # Our classes

X, y = make_classification(n_samples=100, n_features=20, n_classes=2,
                           n_informative=2, n_redundant=0, random_state=24)
# Split features / target into train / validation sets
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=100, stratify=y)

# Build Tree
tree = build_tree_recursive(x_train, y_train)
# Use tree to make prediction
y_pred = predict(tree, x_val)
# Test accuracy of prediction
accuracy = accuracy_score(y_val, y_pred)

print(accuracy)



# label = [1, 0, 1, 1, 0, 1, 2,]
# prob = calc_probs(label)
#
# #user_input = input("Enter choice: ")
# print("Class probabilities: ", prob)
# result = impurity(prob)
# print(result)
