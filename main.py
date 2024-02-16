import time
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree

# clock start
t0 = time.time()

# PROCESS DATA

# TEST 0 - Outlook data set (only contains categorical data)
# df = pd.read_csv('/Users/eaguil/PycharmProjects/p1_randomforests/data/outlook.csv')
# n = len(df.columns) - 1  # Number of features
# trans_ID = df.iloc[:, 0]  # Select IDs
# X = df.iloc[:, 1:n].values  # Select features
# y = df.iloc[:, n].values  # Our classes
# # # # # # # # # # # # # # # # # # # # # # # #

# TEST 1 - BABY data set (sample of original dataset)
# df = pd.read_csv('/Users/eaguil/PycharmProjects/p1_randomforests/data/extra_bb_data.csv')
# selected_rows = df.iloc[:len(df)]

# TEST 2 - FULL data set (can be partitioned to smaller data set using "// n")
df = pd.read_csv("/Users/eaguil/PycharmProjects/p1_randomforests/data/train.csv")
selected_rows = df.iloc[:len(df) // 50]

n = len(df.columns) - 1  # Number of features
trans_ID = selected_rows.iloc[:, 0]  # Select IDs
X_categorical = selected_rows.iloc[:, 1:10]  # Select categorical features
X_numerical = selected_rows.iloc[:, 10:n].values  # Select numerical features
y = selected_rows.iloc[:, n].values  # Our classes

# Imputate data
num_imputer = SimpleImputer(strategy='mean')
X_num_imputed = num_imputer.fit_transform(X_numerical)
cat_imputer = SimpleImputer(strategy='most_frequent')
X_cat_imputed = cat_imputer.fit_transform(X_categorical)

# Encode categorical variables
label_encoders = {}
for i, column in enumerate(X_cat_imputed.T):  # Transpose to iterate over columns
    label_encoders[i] = LabelEncoder()
    X_cat_imputed[:, i] = label_encoders[i].fit_transform(column)

# Concatenate categorical and numerical features
X = np.concatenate((X_num_imputed, X_cat_imputed), axis=1)

# Todo: Add trans_ID to predictions - Since we pass X_test through
#  predict to get our predictions, we just need to attach ID's to
#  attribute values before they get shuffled

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# # MAKE DTs: entropyDT, giniDT, misClassDT
clf = DecisionTree(max_depth=10)

# TRAIN
clf.fit(X_train, y_train)

# PREDICT:
predictions = clf.predict(X_test)

# ACCURACY:
# Building Confusion matrix

confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)

# calculate the True Negative(TN), FalsePositive(FP), FalseNegative (FN) and TruePositive(TP) and put them in an array

TN, FP, FN, TP = confusion_matrix.ravel()
print(TN, FP, FN, TP)

# Calculate True Positive Rate(TPR) and True Negative Rate(TNR)

TPR = TP / (TP + FN)
TNR = TN / (TN + FP)

# Calculate Balanced Accuracy

balanced_accuracy = (TPR + TNR) / 2
print(balanced_accuracy)

t1 = time.time()
print("Time elapsed (in seconds): ", t1 - t0)
