import time
import pandas as pd
import numpy as np
import joblib
import os
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from dt_classifier import DecisionTree
from rt_classifier import RandomForest


time_start = time.time()
# TEST 1 - FULL data set (can be partitioned to smaller data set using "// n")
# Don't forget to replace the file path below with your own path for testing
# df = pd.read_csv("/Users/manjuadhikari/PycharmProjects/p1_randomforests/data/train.csv")

# TRAINING / VALIDATING
df = pd.read_csv("/home/gravy/Desktop/Machine_Learning/project1/p1_randomforests/data/train.csv")

# TESTING
# df = pd.read_csv("/Users/eaguil/PycharmProjects/p1_randomforests/data/test.csv")

selected_rows = df.iloc[:len(df) // 70]
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


# TRAINING/VALIDATING: SPLIT DATA
X_train, X_validation, y_train, y_validation = (
    train_test_split(X, y, test_size=0.2, random_state=1234))

# todo:set stratify=y in train_test_split

# # Using Random Forests (under construction)
# random_forest = RandomForest(ig_type='entropy', node_split_min=10,
#                              max_depth=50, num_features=None, num_trees=10)
# final_prediction = random_forest.build_classifier


# Get the number of samples in the validation set
num_validation_samples = X_validation.shape[0]

# Create an array of indices corresponding to the samples in the validation set
validation_indices = np.arange(len(df))[-num_validation_samples:]

# Retrieve transaction IDs from the original DataFrame using the validation indices
transaction_ids = df.iloc[validation_indices]['TransactionID']


##############################################################################
# # Using Entropy
entropy_DT = DecisionTree(ig_type='entropy', node_split_min=10,
                          max_depth=50, num_features=None)
entropy_DT.fit(X_train, y_train)  # TRAIN
predictions_eDT = entropy_DT.predict(X_validation)  # PREDICT
confMatrix_eDT = metrics.confusion_matrix(y_validation, predictions_eDT)

# Prints predictions to predictions.csv file
predictions = pd.DataFrame({'TransactionID': transaction_ids, 'isFraud': predictions_eDT})
predictions.to_csv("/home/gravy/Desktop/Machine_Learning/project1/p1_randomforests/data/predictions.csv", index=False)

print(confMatrix_eDT)
TN = confMatrix_eDT[0, 0]
FP = confMatrix_eDT[0, 1]
FN = confMatrix_eDT[1, 0]
TP = confMatrix_eDT[1, 1]
print(TN, FP, FN, TP)
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
balanced_accuracy = (TPR + TNR) / 2
print("Using Entropy, our balanced accuracy is: ", balanced_accuracy)



# This section tries to save the tree running entropy
filename = 'train_model.sav'
joblib.dump(entropy_DT, filename)

loaded_model = joblib.load('train_model.sav')

# Print the loaded model
print(loaded_model)

# Step 1: Confirm the type of entropy_DT
print(type(entropy_DT))

# Step 2: Check the type of the loaded model
print(type(loaded_model))

# Step 3: Check if the file exists
print(os.path.exists('train_model.sav'))


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
