import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from testProgram.DecisionTree import DecisionTree

# Read in the data
# df = pd.read_csv('extra_bb_data.csv')  # small sample of our original dataset
df = pd.read_csv('/Users/eaguil/PycharmProjects/p1_randomforests/data/train.csv')

selected_rows = df.iloc[:len(df)//50]

# Split the data into tID / features / target
tID = selected_rows.iloc[:, 0].values  # don't know if this is really necessary?
X = selected_rows.iloc[:, 1:25].values  # ALL features
y = selected_rows.iloc[:, 26].values  # Our classes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=100)

# data = datasets.load_breast_cancer()
# X, y = data.data, data.target
# print(data.feature_type)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234)


clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, predictions)
print(acc)
