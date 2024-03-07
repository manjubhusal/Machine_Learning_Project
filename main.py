import time
import pandas as pd
from sklearn.model_selection import train_test_split
from dt_classifier import DecisionTree
from rt_classifier import RandomForest
from helper_functions import *

# Program Version 3.5

time_start = time.time()

# TRAINING / VALIDATING
# df = pd.read_csv("/Users/eaguil/PycharmProjects/p1_randomforests/data/train.csv")
# df = pd.read_csv("/Users/manjuadhikari/PycharmProjects/p1_randomforests/data/train.csv")
df = pd.read_csv("C:/Users/Ester/PycharmProjects/p1_randomforests/data/train.csv")
# df = pd.read_csv("/nfs/student/e/eaguilera/p1_randomforests/data/train.csv")
# df = pd.read_csv("/home/gravy/Desktop/Machine_Learning/project1/p1_randomforests/data/train.csv")

X, y, train_trans_ID = process_data(df, "train")  # We won't actually need or use these trans_IDs
X_train, X_validation, y_train, y_validation = (
    train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y))

# TESTING
df_test = pd.read_csv("C:/Users/Ester/PycharmProjects/p1_randomforests/data/test.csv")
X_test, y_empty, test_trans_ID = process_data(df_test, "test")  # y_test is empty and won't be used


##############################################################################
# Set info gain type / Impurity for either DT or RF classifier
impurity_type = 'entropy'

# # Decision Tree Classifier
dt_model = DecisionTree(X_train, y_train, ig_type=impurity_type,
                        node_split_min=10, max_depth=45, num_features=5)  # TRAIN
dt_prediction = dt_model.predict(X_train)  # PREDICT
accuracy = calc_balanced_accuracy(y_validation, dt_prediction)  # MEASURE ACCURACY
print("Decision Tree Classifier: Using ",
      impurity_type, " our balanced accuracy is: ", accuracy)

# # Random Forests Classifier
# random_forest = RandomForest(ig_type=impurity_type, node_split_min=10,
#                              max_depth=45, num_features=5, num_trees=1)
# rf_prediction = random_forest.build_classifier(X_train, y_train, X_test)
# accuracy = calc_balanced_accuracy(y_validation, rf_prediction)  # MEASURE ACCURACY
# print("Random Forest Classifier: Using ",
#       impurity_type, " our balanced accuracy is: ", accuracy)
##############################################################################
# # Code for adding trans_ID to predictions

# Prints predictions to rf_predictions.csv file for Random Forest
# predictions_rf = pd.DataFrame({'TransactionID': test_trans_ID, 'isFraud': rf_prediction})
# # predictions_rf.to_csv("/home/gravy/Desktop/Machine_Learning/project1/"
# #                       "p1_randomforests/data/rf_predicitons.csv", index=False)
# predictions_rf.to_csv("C:/Users/Ester/Desktop/rf_predictions/rf_pred.csv", index=False)

##############################################################################
# # Code for saving trees
# This section tries to save the tree running entropy
# filename = 'train_model.sav'
# joblib.dump(entropy_DT, filename)
#
# loaded_model = joblib.load('train_model.sav')
#
# # Print the loaded model
# print(loaded_model)
#
# # Step 1: Confirm the type of entropy_DT
# print(type(entropy_DT))
# # Step 2: Check the type of the loaded model
# print(type(loaded_model))
#
# # Step 3: Check if the file exists
# print(os.path.exists('train_model.sav'))
##############################################################################

time_end = time.time()
time_elapsed = time_end - time_start
print("Total runtime (in seconds): ", time_elapsed)
