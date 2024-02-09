import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read in the data
df = pd.read_csv('data/train.csv')

# Split the data into tID / features / target
tID = df.iloc[:, 0].values  # don't know if this is really necessary
features = df.iloc[:, 1:25].values  # ALL features
target = df.iloc[:, 26].values  # last column

# Split features / target into
feat_train, feat_test, test_train, test_test = train_test_split(
    features, target, random_state=100)

# Step 2: Calculate impurity - calculate_impurity(dataSet)
# type - type of impurity
# type (1) = entropy, type (2) = mis. error, type (3) = gini index
# calculate_impurity(dataSet, type)
# # # calculate entropy
# # # calculate mis. error
# # # calculate gini index


# Step 1: Read in the data

"""
# Step 2: Calculate impurity - calculate_impurity(dataSet)
# This portion of code calculates the impurity (I think) of a 
# given simple array. I correctly calculates, entropy, miss-classification error
# and the gini index. After the chunks are processed, you'll be prompted to enter
# a choice, e,g, or m. This will definitely be changed later.
"""
#def calc_info_gain(dataSet, )


def calc_probs(label):
    total_instances = len(label)
    class_count = {}

    for l in label:
        if l in class_count:
            class_count[l] += 1
        else:
            class_count[l] = 1

    prob = [count / total_instances for count in class_count.values()]
    return prob


def impurity(prob):

    e_prob = np.array(prob)
    entropy = -np.sum(prob * np.log2(e_prob))

    g_array = np.array(prob)
    gini = -np.sum(g_array ** 2)

    miss_array = np.array(prob)
    miss_error = - np.max(miss_array)

    final = max(entropy, gini, miss_error)

    if (final == entropy):
        print("Entropy is largest: ")

    elif (final == gini):
        print("Gini is larget: ")

    else:
        print("Miss_Error is larges: ")

    return final



label = [1, 0, 1, 1, 0, 1, 2,]
prob = calc_probs(label)

#user_input = input("Enter choice: ")
print("Class probabilities: ", prob)
result = impurity(prob)
print(result)

# Step 3: Calculate information gain - calculate_infoGain(dataSet, att)

# Step 4: Choose the best IG as split criterium - choose_splitCriterium()

# Step 5: Build Decision Tree / Random Forest

# Step 6: Experiment 1

# Step 7: Experiment 2


# Step 3: Calculate information gain - calculate_infoGain(dataSet, att)
# IG(D, A)_entropy = calculate_impurity(D, 1) - ...
# IG(D, A)_ME = calculate_impurity(D, 2) - ...
# IG(D, A)_GI = calculate_impurity(D, 3) - ...

# Step 4: Choose the best IG as split criterium - choose_splitCriterium()

# Step 5: Build Decision Tree / Random Forest (Class)


# Step 6: Experiment 1


# Step 7: Experiment 2
