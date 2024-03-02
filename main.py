import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from random_forest import RandomForest

# PROGRAM VERSION 2.0 - changes have been recorded in README

time_start = time.time()
# TEST 1 - FULL data set (can be partitioned to smaller data set using "// n")
# Don't forget to replace the file path below with your own path for testing
df = pd.read_csv("/Users/manjuadhikari/PycharmProjects/p1_randomforests/data/train.csv")
selected_rows = df.iloc[:len(df)// 5]

n = len(df.columns) - 1  # Number of features
trans_ID = selected_rows.iloc[:, 0]  # Select IDs
X_categorical = selected_rows.iloc[:, 1:10].values  # Select categorical features
X_numerical = selected_rows.iloc[:, 10:n].values  # Select numerical features
y = selected_rows.iloc[:, n].values  # Our classes

num_impute = SimpleImputer(strategy='mean')
X_num_imputed = num_impute.fit_transform(X_numerical)
cat_impute = SimpleImputer(strategy='most_frequent')
X_cat_imputed = cat_impute.fit_transform(X_categorical)

label_encoders = {}
for i, column in enumerate(X_cat_imputed.T):  # Transpose to iterate over columns
    label_encoders[i] = LabelEncoder()
    X_cat_imputed[:, i] = label_encoders[i].fit_transform(column)

X = np.concatenate((X_cat_imputed, X_num_imputed), axis=1)

# Todo: Add trans_ID to predictions - Since we pass X_test through
#  predict to get our predictions, we just need to attach ID's to
#  attribute values before they get shuffled

# SPLIT DATA
X_train, X_validation, y_train, y_validation = (
    train_test_split(X, y, test_size=0.2, random_state=1234))

# todo:set stratify=y in train_test_split

##############################################################################
# # Using Entropy
entropy_DT = DecisionTree(ig_type='entropy', node_split_min=10,
                          max_depth=50, num_features=None)
entropy_DT.fit(X_train, y_train)  # TRAIN
predictions_eDT = entropy_DT.predict(X_validation)  # PREDICT
# clf = RandomForest(max_depth=10)
# clf.fit(X_train, y_train)
# predictions_eDT = clf.predict(X_test)
confMatrix_eDT = metrics.confusion_matrix(y_validation, predictions_eDT)
print(confMatrix_eDT)
# TN, FP, FN, TP = confMatrix_eDT.ravel()
TN = confMatrix_eDT [0,0]
FP = confMatrix_eDT [0,1]
FN = confMatrix_eDT [1,0]
TP = confMatrix_eDT [1,1]
print(TN, FP, FN, TP)
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
balanced_accuracy = (TPR + TNR) / 2
print("Using Entropy, our balanced accuracy is: ", balanced_accuracy)
##############################################################################
# # Using Gini impurity
# gini_DT = DecisionTree(max_depth=10, ig_type='gini')
# gini_DT.fit(X_train, y_train)  # TRAIN
# predictions_gDT = gini_DT.predict(X_validation)  # PREDICT
# confMatrix_gDT = metrics.confusion_matrix(y_validation, predictions_gDT)
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
# predictions_mDT = misClass_DT.predict(X_validation)  # PREDICT
# confMatrix_mDT = metrics.confusion_matrix(y_validation, predictions_mDT)
# print(confMatrix_mDT)
# TN, FP, FN, TP = confMatrix_mDT.ravel()
# print(TN, FP, FN, TP)
# TPR = TP / (TP + FN)
# TNR = TN / (TN + FP)
# balanced_accuracy = (TPR + TNR) / 2
# print("Using misclassification error, our balanced accuracy is: ", balanced_accuracy)
###############################################################################

time_end = time.time()
time_elapsed = time_end - time_start
print("Total runtime (in seconds): ", time_elapsed)
