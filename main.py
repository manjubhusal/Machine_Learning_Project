import random
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from dt_classifier import DecisionTree
from rt_classifier import RandomForest
from helperlib import *


# Program Version 4.0
def main():

    config = read_config("configuration_files/config1.yaml")
    df_train_val = pd.read_csv("data/input files/train.csv")
    df_test = pd.read_csv("data/input files/test.csv")

    # Access specific configurations
    action_type = config['action_type']
    model_type = config['model_type']
    ig_type = config['ig_type']
    alpha_level = float(config['alpha_level'])
    min_sample_split = config['min_sample_split']
    max_depth = config['max_depth']
    num_features = config['num_features']
    num_trees = config['num_trees']

    # We won't actually need or use train_trans_IDs
    X, y, train_trans_ID = process_data(df_train_val, "validate")
    X_train, X_validation, y_train, y_validation = (
        train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y))

    # Needed for final report but obviously not calculated when action_type==test
    accuracy = None

    # Unique ID to label each kaggle run - this allows us uniquely label output
    # files based on the run ID, so run #123 is kaggle123.csv. Because this doesn't
    # store all of our IDs ever created, we may get duplicate IDs from past runs.
    # All runs MUST be recorded on the spreadsheet where you will input run ID; if
    # you get a run ID that's been used before, simply make note of this on the
    # spreadsheet.
    run_ID = random.randint(1, 10000)

    # Conditional logic based on configuration
    # TRAIN CLASSIFIER -> PREDICT -> WRITE OUTPUT TO CSV
    if action_type == "test":
        print("Running in test mode...\n")
        X_test, y_empty, test_trans_ID = process_data(df_test, action_type)
        if model_type == "decision_tree":
            dt_model = DecisionTree(X_train, y_train, ig_type, alpha_level, min_sample_split, max_depth, num_features)
            test_dt_prediction = dt_model.predict(X_test)
            df_dt_pred = pd.DataFrame({'TransactionID': test_trans_ID, 'isFraud': test_dt_prediction})
            filename = f"data/output files/kaggle_{run_ID}.csv"
            df_dt_pred.to_csv(filename, index=False)
        else:
            rt_model = RandomForest(ig_type, alpha_level, min_sample_split, max_depth, num_features, num_trees)
            test_rf_prediction = rt_model.build_classifier(X_train, y_train, X_test)
            df_rf_pred = pd.DataFrame({'TransactionID': test_trans_ID, 'isFraud': test_rf_prediction})
            filename = f"data/output files/kaggle_{run_ID}.csv"
            df_rf_pred.to_csv(filename, index=False)
    # TRAIN CLASSIFIER -> PREDICT -> VALIDATE -> MEASURE ACCURACY
    elif action_type == "validate":
        print("Running in validate mode...\n")
        if model_type == "decision_tree":
            dt_model = DecisionTree(X_train, y_train, ig_type, alpha_level, min_sample_split, max_depth, num_features)
            val_dt_prediction = dt_model.predict(X_validation)
            accuracy = calc_balanced_accuracy(y_validation, val_dt_prediction)
        else:
            rf_model = RandomForest(ig_type, alpha_level, min_sample_split, max_depth, num_features, num_trees)
            val_rf_prediction = rf_model.build_classifier(X_train, y_train, X_validation)
            accuracy = calc_balanced_accuracy(y_validation, val_rf_prediction)
    else:
        print("action_type error in main")

    print_run_report(config, accuracy, run_ID)


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    time_elapsed = time_end - time_start
    print("Runtime (secs): ", time_elapsed)
