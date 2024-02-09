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
feat_train, feat_test, target_train, target_test = train_test_split(
    features, target, random_state=100)


# def calc_info_gain(dataSet, )

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




