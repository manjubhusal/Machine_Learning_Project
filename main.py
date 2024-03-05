import time
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
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

##############################################################################
# # Decision Tree Classifier
impurity_type = 'entropy'
dt_model = DecisionTree(X_train, y_train, ig_type=impurity_type,
                        node_split_min=5, max_depth=15, num_features=5)  # TRAIN
dt_prediction = dt_model.predict(X_validation)  # PREDICT
accuracy = calc_balanced_accuracy(y_validation, dt_prediction)  # MEASURE ACCURACY
print("Using ", impurity_type, " our balanced accuracy is: ", accuracy)

# # Random Forests Classifier
# random_forest = RandomForest(ig_type=impurity_type, node_split_min=10,
#                              max_depth=50, num_features=None, num_trees=10)
# rf_prediction = random_forest.build_classifier(X_train, y_train, X_validation)

# # This section tries to save the tree running entropy
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
#
# # Step 2: Check the type of the loaded model
# print(type(loaded_model))
#
# # Step 3: Check if the file exists
# print(os.path.exists('train_model.sav'))


time_end = time.time()
time_elapsed = time_end - time_start
print("Total runtime (in seconds): ", time_elapsed)
