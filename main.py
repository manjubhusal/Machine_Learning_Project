import time
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from p1_randomforests.DecisionTree import DecisionTree

# clock start
t0 = time.time()

# Read in the data

# TEST 1 - BABY data set (sample of original dataset)
# df = pd.read_csv('C:/Users/Ester/PycharmProjects/p1_randomforests/data/extra_bb_data.csv')
# selected_rows = df.iloc[:len(df)]
# #

# TEST 2 - FULL data set (can be partitioned to smaller data set using "// n")
df = pd.read_csv("C:/Users/Ester/PycharmProjects/p1_randomforests/data/train.csv")
selected_rows = df.iloc[:len(df)]
# #

X_categorical = selected_rows.iloc[:, 1:25]  # Select categorical features
X_numerical = selected_rows.iloc[:, 25:26].values  # Select numerical features
y = selected_rows.iloc[:, 26].values  # Our classes

# Handle missing values
X_numerical = np.nan_to_num(X_numerical)  # Replace NaN values with zero, you can replace with other values if needed

# Encode categorical variables
label_encoders = {}
for column in X_categorical.columns:
    label_encoders[column] = LabelEncoder()
    X_categorical[column] = label_encoders[column].fit_transform(X_categorical[column])

# Concatenate categorical and numerical features
X = np.concatenate([X_categorical, X_numerical], axis=1)

# Handle infinite values
large_number = 1e9
X[np.abs(X) > large_number] = np.sign(X[np.abs(X) > large_number]) * large_number

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, predictions)
print(acc)

t1 = time.time()
print("Time elapsed (in seconds): ", t1 - t0)
