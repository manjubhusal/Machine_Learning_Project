# P1_RandomForests
## How to run the program
Include steps on how to run the program for the 
testers and where to specify path to the
dataset.
1. Make sure environment is appropriately set up. 
2. In **configuration_files** directory, adjust parameters to
desired setting in config1.yaml file. The values you chose must 
follow these guidelines otherwise you will get a **ValueError**:
- action_type must be one of the following: ["test", "validate"]
- model_type must be one of the following: ["decision_tree", "random_forest"]
- ig_type must be one of the following: ["entropy", "mis_error", "gini"]
- alpha_level must be a float in the range [0.01 - 0.9]
- min_sample_split must be an int in the range [2 - 40]
- max_depth must be an int in the range [1 - 100]
- num_features must be an int in the range [1 - 20]
- num_trees must be an int in the range [1 - 50] 
***(set to 1 if model_type=decision_tree)*** 
3. **IMPORTANT!!**: to alternate between using the entire
dataset and just a fraction of it, please manually change
this portion of the code inside the *process_data* function
in the *helperlib.py* file: `selected_rows = df.iloc[:len(df) // 70]`
4. Make sure to *save* your changes to the config file, once you've
done this, you only need to run the main.py file as normal and the
program will tell you what action you are running early on and
once the program finishes it will print out a program run report
to the terminal which will verify the config settings you ran
the program runtime, and if applicable, it will output accuracy.
***

- Kaggle API 1.6.5 (installation below)
  - Go to https://www.kaggle.com/settings/account and "create a new API token".
  - This will download a kaggle.json file you will need. 
  - Commands below are for Mac/Linux but process should be similar for Windows.
  - Line 1 installs kaggle; if you get permission errors, try: ```pip install --user kaggle```.
  - Line 2 is to move the kaggle.json file you downloaded to ~/.kaggle/kaggle.json.
  - Line 3 is to ensure that other users of your computer do not have read access to your credentials.
  - Line 4 is to ensure your have the correct API version.
```commandline
pip install kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle --version
```

**To install or add packages on Pycharm**, go to Pycharm's
settings and type in "interpreter"; select "python interpreter"
which should show you a list of your currently added python packages.
Here you can install any packages your missing. Some packages may already 
installed and only need to be added.

## Project goal

Code the following from scratch:
- Decision tree that uses IG as split criterium and chi square as alternate 
termination rule.
- Implement Information Gain (IG) with: 1. Entropy, 2. Gini Index, 3. Misclassification Error
- Implement a random forest based on your decision tree and any criteria 
of your choosing.




