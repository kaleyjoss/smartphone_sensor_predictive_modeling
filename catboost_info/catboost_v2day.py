
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import json 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, make_scorer, confusion_matrix, mean_squared_error, mean_absolute_error

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier
from catboost import Pool



# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.getcwd()))
# Add project root to sys.path
sys.path.append(project_root)
# Define data directory
brighten_dir = '/Users/demo/Library/CloudStorage/Box-Box/Holmes_lab_kaley/motif_proj/smartphone_sensor_predictive_modeling/BRIGHTEN_data'

# Import and reload my custom scripts
from scripts import preprocessing as pre
from scripts import visualization as vis
from scripts import feature_selection as fs
from scripts import clustering as cl
importlib.reload(pre)
importlib.reload(vis)
importlib.reload(fs)
importlib.reload(cl)

# Import from cloned github repos
import hyperopt
print(hyperopt.__file__)
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
import hyperopt.pyll.stochastic
################ DEFINE column variables from data ###################
from scripts.variables import id_columns, daily_cols_v1, daily_v2_common 


## Load in dfs scaled
df_names = ['v2_day']
df_pca = ['v1_day_pca', 'v1_week_pca']
df_all = df_names + df_pca
results={}

for name in df_names:
    print(f"--- Dataset: {name} ---")
    X_train_unclean = pd.read_csv(os.path.join(brighten_dir, f'{name}_X_train.csv'))
    y_train_unclean = pd.read_csv(os.path.join(brighten_dir, f'{name}_y_train.csv'))
    # test
    X_test_unclean = pd.read_csv(os.path.join(brighten_dir, f'{name}_X_test.csv'))
    y_test_unclean = pd.read_csv(os.path.join(brighten_dir, f'{name}_y_test.csv'))

    X_train = X_train_unclean.drop(columns=[col for col in X_train_unclean.columns if col in id_columns or 'Unnamed' in col])
    y_train = y_train_unclean.drop(columns=[col for col in y_train_unclean.columns if 'Unnamed' in col]).squeeze()
    X_test = X_test_unclean.drop(columns=[col for col in X_test_unclean.columns if col in id_columns or 'Unnamed' in col])
    y_test = y_test_unclean.drop(columns=[col for col in y_test_unclean.columns if 'Unnamed' in col]).squeeze()

    # Define parameter grid based on dataset
    if name == "v1_day":
        param_grid = {
                'iterations': [i for i in range(140, 170)],
                'learning_rate': [round(i, 3) for i in uniform.rvs(loc=0.15, scale=0.25, size=10)],
                'depth': [i for i in range(8, 12)],
                'l2_leaf_reg': [round(i, 3) for i in uniform.rvs(loc=7, scale=9, size=10)],
                'border_count': [i for i in range(110, 131)],
                'random_strength': [round(i, 3) for i in uniform.rvs(loc=0.4, scale=0.2, size=10)]
            }
    elif name == "v2_day":
        param_grid = {
            'iterations': [i for i in range(270, 296)],
            'learning_rate': [round(i, 5) for i in uniform.rvs(loc=0.01, scale=0.015, size=10)],
            'depth': [i for i in range(2, 6)],
            'l2_leaf_reg': [round(i, 3) for i in uniform.rvs(loc=3, scale=2, size=10)],
            'border_count': [i for i in range(110, 131)],
            'random_strength': [round(i, 3) for i in uniform.rvs(loc=0.1, scale=0.25, size=10)]
        }
    elif name == "v1_week":
        param_grid = {
            'iterations': [i for i in range(278, 296)],
            'learning_rate': [round(i, 3) for i in uniform.rvs(loc=0.18, scale=0.09, size=10)],
            'depth': [i for i in range(4, 8)],
            'l2_leaf_reg': [round(i, 3) for i in uniform.rvs(loc=9, scale=2, size=10)],
            'border_count': [i for i in range(150, 166)],
            'random_strength': [round(i, 3) for i in uniform.rvs(loc=0.58, scale=0.17, size=10)]
        }
    # Initialize the CatBoostClassifier
    model = CatBoostClassifier(random_state=42, verbose=0)

    # Perform grid search
    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    # Log best results
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    results[name] = {"best_params": best_params}

    # Save grid search results to a JSON file
    with open(f"{name}_catboost_gridsearch_results.json", "w") as f:
        json.dump(results, f, indent=4)

    final_model = CatBoostClassifier(**best_params, random_state=42)

    final_model.fit(X_train, y_train)
    score = final_model.score(X_test, y_test)

    # Optionally, save predictions to a file
    with open(f"{name}_predictions.json", "w") as f:
        json.dump({"accuracy": score}, f, indent=4)

    print(f"{name} accuracy: {score:.4f}")
