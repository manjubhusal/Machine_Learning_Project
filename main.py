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


# Step 1: Read in the data

"""
# Step 2: Calculate impurity - calculate_impurity(dataSet)
# This portion of code calculates the impurity (I think) of a 
# given simple array. I correctly calculates, entropy, miss-classification error
# and the gini index. After the chunks are processed, you'll be prompted to enter
# a choice, e,g, or m. This will definitely be changed later.
"""
def entropy(prob):
    prob = np.array(prob)
    return -np.sum(prob * np.log2(prob))


def gini_index(prob):
    prob_array = np.array(prob)
    return 1 - np.sum(prob_array ** 2)


def miss_class_err(prob):
    prob_array = np.array(prob)
    return 1 - np.max(prob_array)


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


def impurity(prob, choice):
    match choice:
        case 'e':
            print("Impurity using Entropy: ")
            return entropy(prob)
        case 'g':
            print("Impurity using Gini Index: ")
            return gini_index(prob)
        case 'm':
            print("Impurity using Miss-Class Error: ")
            return miss_class_err(prob)
        case _:
            return "Invalid Choice"


label = ['a', 'b', 'a', 'b', 'c', 'a']
prob = calc_probs(label)

user_input = input("Enter choice: ")
print("Class probabilities: ", prob)
result = impurity(prob, user_input)
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
