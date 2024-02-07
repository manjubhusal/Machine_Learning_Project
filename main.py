import pandas as pd
import numpy as np
import scipy
import csv

# Step 1: Read in the data
num_rows = 0
df = pd.DataFrame

for df_Chunk in pd.read_csv("data/train.csv", chunksize=10000):
    num_rows += len(df_Chunk)  # find total number of rows after each df chunk
    print("processed {0}".format(num_rows))  # test code
    dataSet = df_Chunk
    # df = df.append(calculate_infoGain(df_chunk, att))




# Step 2: Calculate impurity - calculate_impurity(dataSet)
# type - type of impurity
# type (1) = entropy, type (2) = mis. error, type (3) = gini index
# calculate_impurity(dataSet, type)
# # # calculate entropy
# # # calculate mis. error
# # # calculate gini index


# Step 3: Calculate information gain - calculate_infoGain(dataSet, att)
# IG(D, A)_entropy = calculate_impurity(D, 1) - ...
# IG(D, A)_ME = calculate_impurity(D, 2) - ...
# IG(D, A)_GI = calculate_impurity(D, 3) - ...

# Step 4: Choose the best IG as split criterium - choose_splitCriterium()

# Step 5: Build Decision Tree / Random Forest (Class)


# Step 6: Experiment 1


# Step 7: Experiment 2
