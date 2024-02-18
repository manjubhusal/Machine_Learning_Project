import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.impute import SimpleImputer
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
df = pd.read_csv("C:/Users/Ester/PycharmProjects/p1_randomforests/data/train.csv")
selected_rows = df.iloc[:len(df) // 20]

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


# MAKE DTs -> TRAIN -> PREDICT -> TEST ACCURACY
# 1. Calculate the True Negative(TN), FalsePositive(FP), FalseNegative (FN)
#    and TruePositive(TP) and put them in an array
# 2. Calculate True Positive Rate(TPR) and True Negative Rate(TNR)
# 3. Calculate Balanced Accuracy


# Using Entropy
entropy_DT = DecisionTree(max_depth=10, ig_type='entropy')  # MAKE DT
entropy_DT.fit(X_train, y_train)  # TRAIN
predictions_eDT = entropy_DT.predict(X_test)  # PREDICT
confMatrix_eDT = metrics.confusion_matrix(y_test, predictions_eDT)
print(confMatrix_eDT)
TN, FP, FN, TP = confMatrix_eDT.ravel()
print(TN, FP, FN, TP)
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
balanced_accuracy = (TPR + TNR) / 2
print("Using Entropy, our balanced accuracy is: ", balanced_accuracy)
##############################################################################
# # Using Gini impurity
# gini_DT = DecisionTree(max_depth=10, ig_type='gini')
# gini_DT.fit(X_train, y_train)  # TRAIN
# predictions_gDT = gini_DT.predict(X_test)  # PREDICT
# confMatrix_gDT = metrics.confusion_matrix(y_test, predictions_gDT)
# print(confMatrix_gDT)
# TN, FP, FN, TP = confMatrix_gDT.ravel()
# print(TN, FP, FN, TP)
# TPR = TP / (TP + FN)
# TNR = TN / (TN + FP)
# balanced_accuracy = (TPR + TNR) / 2
# print("Using Gini Impurity, our balanced accuracy is: ", balanced_accuracy)
###############################################################################
# # Using misclassification error
# misClass_DT = DecisionTree(max_depth=10, ig_type='mis_error')
# misClass_DT.fit(X_train, y_train)  # TRAIN
# predictions_mDT = misClass_DT.predict(X_test)  # PREDICT
# confMatrix_mDT = metrics.confusion_matrix(y_test, predictions_mDT)
# print(confMatrix_mDT)
# TN, FP, FN, TP = confMatrix_mDT.ravel()
# print(TN, FP, FN, TP)
# TPR = TP / (TP + FN)
# TNR = TN / (TN + FP)
# balanced_accuracy = (TPR + TNR) / 2
# print("Using Gini Impurity, our balanced accuracy is: ", balanced_accuracy)
###############################################################################


t1 = time.time()
print("Time elapsed (in seconds): ", t1 - t0)
