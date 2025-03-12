from collections import Counter
from scipy.stats import chi2
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import yaml


def print_run_report(config, accuracy, run_ID):
    # Header
    print("Program Run Report\n" + "=" * 20)
    # Configurations
    print("Configuration Settings:")
    for key, value in config.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    # Spacer
    print("-" * 20)
    # Run ID
    print("Run ID: ", str(run_ID))
    # Accuracy
    print("Accuracy: ", accuracy)
    print("-" * 20 + "\n")


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Validate config
    valid_action_types = ["test", "validate"]
    valid_model_types = ["decision_tree", "random_forest"]
    valid_ig_types = ["entropy", "mis_error", "gini"]

    action_type = config.get('action_type')
    model_type = config.get('model_type')
    ig_type = config.get('ig_type')
    alpha_level = float(config.get('alpha_level'))
    min_sample_split = config.get('min_sample_split')
    max_depth = config.get('max_depth')
    num_features = config.get('num_features')
    num_trees = config.get('num_trees')

    if action_type not in valid_action_types:
        raise ValueError(f"Invalid action_type: {action_type}")
    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model_type: {model_type}")
    if ig_type not in valid_ig_types:
        raise ValueError(f"Invalid ig_type: {ig_type}")
    if not 0.01 <= alpha_level <= 0.9:
        raise ValueError("alpha_level must be a float in the range [0.01 - 0.9]")
    if not 2 <= min_sample_split <= 40:
        raise ValueError("min_sample_split must be an int in the range [2 - 40]")
    if not 1 <= max_depth <= 100:
        raise ValueError("max_depth must be an int in the range [1 - 100]")
    if not 1 <= num_features <= 20:
        raise ValueError("num_features must be an int in the range [1 - 20]")
    if not 1 <= num_trees <= 50:
        raise ValueError("num_trees must be an int in the range [1 - 50]")

    return config


def process_data(df, mode):
    # n = len(df.columns) - 1  # Number of features
    n = 26
    selected_rows = df.iloc[:len(df)]

    if mode == "validate":
        y = selected_rows.iloc[:, n].values  # Our classes
    else:
        y = None

    trans_ID = selected_rows.iloc[:, 0]  # Select IDs
    X_categorical = selected_rows.iloc[:, 1:10].values  # Select categorical features
    X_numerical = selected_rows.iloc[:, 10:n].values  # Select numerical features

    # Accounting for missing data
    num_impute = SimpleImputer(strategy='mean')
    X_num_imputed = num_impute.fit_transform(X_numerical)
    cat_impute = SimpleImputer(strategy='most_frequent')
    X_cat_imputed = cat_impute.fit_transform(X_categorical)

    label_encoders = {}
    for i, column in enumerate(X_cat_imputed.T):  # Transpose to iterate over columns
        label_encoders[i] = LabelEncoder()
        X_cat_imputed[:, i] = label_encoders[i].fit_transform(column)

    X = np.concatenate((X_cat_imputed, X_num_imputed), axis=1)

    return X, y, trans_ID


def calc_balanced_accuracy(y_true, y_pred):
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TP = conf_matrix[1, 1]
    print(TN, FP, FN, TP)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    balanced_accuracy = (TPR + TNR) / 2
    return balanced_accuracy


def calc_gini(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    gini = 1 - sum(np.square(ps))
    return gini


def calc_misclass_error(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    miss_error = 1 - np.max(ps)
    return miss_error


def calc_entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum(ps * np.log(ps + 1e-12))


def chi_square(obs):
    total = np.sum(obs)
    expec = np.full_like(obs, total / len(obs))
    stat = np.sum((obs - expec) ** 2 / expec)
    df = len(obs) - 1
    p_val = 1 - chi2.cdf(stat, df)
    return stat, p_val


def should_split(attribute_values, class_labels, alpha_level):
    # Perform chi-square test
    attribute_values = np.array(attribute_values)
    class_labels = np.array(class_labels)
    observed_freq = np.histogram2d(attribute_values, class_labels)[0]
    chi_2, p = chi_square(observed_freq)

    # Check if any attribute is significantly correlated with the class
    if p.any() > alpha_level:  # Adjust significance level as needed
        return False  # Stop splitting
    else:
        return True  # Continue splitting


# todo: ESTER -> add comments to this
def calc_info_gain(impurity, y, selected_feature, threshold):

    left_idxs = np.where(selected_feature <= threshold)[0]
    right_idxs = np.where(selected_feature > threshold)[0]

    n_left = len(left_idxs)
    n_right = len(right_idxs)
    n = len(y)

    if n_left == 0 or n_right == 0:
        return 0

    if impurity == 'entropy':
        parent_impurity = calc_entropy(y)
        e_left = calc_entropy(y[left_idxs])
        e_right = calc_entropy(y[right_idxs])
        child_impurity = (n_left / n) * e_left + (n_right / n) * e_right
        information_gain = parent_impurity - child_impurity
    elif impurity == 'gini':
        parent_impurity = calc_gini(y)
        g_left, g_right = calc_gini(y[left_idxs]), calc_gini(y[right_idxs])
        child_impurity = (n_left / n) * g_left + (n_right / n) * g_right
        information_gain = parent_impurity - child_impurity
    elif impurity == 'mis_error':
        parent_impurity = calc_misclass_error(y)
        m_left, m_right = (calc_misclass_error(y[left_idxs]),
                           calc_misclass_error(y[right_idxs]))
        child_impurity = (n_left / n) * m_left + (n_right / n) * m_right
        information_gain = parent_impurity - child_impurity
    else:
        print("INFO GAIN CALC ERROR")

    return information_gain


# todo: ESTER -> try to see if we can use a different way of
#  doing representative_class - maybe without using Counter
def representative_class(y):
    # In the case of our dataset, the majority class is '0' so that is what
    # we are returning
    if len(y) == 0:
        return 0
    counter = Counter(y)
    most_common_element = counter.most_common(1)
    if most_common_element:  # Check if the list is not empty
        return most_common_element[0][0]
    else:
        return 0
