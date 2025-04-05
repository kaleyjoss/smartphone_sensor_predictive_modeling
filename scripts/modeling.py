### Modeling functions
import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

############ LOAD in PACKAGES  #############
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import roc_auc_score


############ Individual Random Forest Walk-forward Validation & TT Split Functions ##############
# split a univariate dataset into train/test sets
# based on a percentage of rows for test dataset by 'test_prc' (for example, 0.2)
def train_test_split(data, n_test, verbose=False):
	if verbose:
		print(f'train_test_split: Train rows = {data.shape[0] - n_test}; Test rows = {n_test}')
	
 # If data is already a NumPy array, use normal slicing
	return data[:n_test, :], data[n_test:, :]




from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# Decision tree Classifier
def dt_regressor(df, X_cols, y_col):
    print("X cols:",X_cols)
    y_cols = y_col
    df = df[X_cols + y_cols].dropna()
    print(df.shape)
    X = df[X_cols]
    y = df[y_cols]
    print(f'df shape {df.shape}, X {X.shape}, y {y.shape}, {y_cols}')

    # Assuming X and y are already defined
    # Step 1: Split off the held-out validation set (15%)
    X_train_test, X_val, y_train_test, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # Step 2: Split the remaining 85% into training (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.2, random_state=42)

    # Print dataset shapes
    print('Check for set overlap:',set(X_train.index) & set(X_test.index), set(X_val.index) & set(X_test.index), set(X_train.index) & set(X_val.index))  # Should return an empty set
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Range {y_test.min()} to {y_test.max()}")

    # Decision tree
    regressor = DecisionTreeRegressor(criterion='friedman_mse', max_depth=2, min_samples_split=4, random_state=42)

    # Cross validations
    cv = KFold(n_splits=5, shuffle=True, random_state=40)
    cross_val_results = cross_val_score(regressor, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

    # Print CV results
    print(f"Cross-validation MSE scores: {-cross_val_results}")
    print(f"Average Cross-validation MSE scores: {-cross_val_results.mean()}")

    # Train and fit the model
    regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Regression mean_absolute_error: {mae:.2f}, r2: {r2:.2f}')


    # # Test AUC-ROC
    # y_pred_prob = regressor.predict_proba(X_test)[:, 1]  # Get probability of class 1
    # auc = roc_auc_score(y_test, y_pred_prob)
    # print(f'AUC-ROC Score: {auc:.2f}')

    # Find optimal parameters
    optimized_reg = DecisionTreeRegressor(max_depth=2, min_samples_split=4, criterion='friedman_mse')
    optimized_reg.fit(X_train, y_train)

    # Try optimized classifier on held-out validation set
    y_pred_val = optimized_reg.predict(X_val)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Regression Held-out mean_absolute_error: {mae:.2f}, r2: {r2:.2f}')

    # Plot figure
    plt.figure(figsize=(12, 8))
    tree.plot_tree(optimized_reg, feature_names=X.columns, filled=True)
    plt.show()

    return mae, r2






from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split

def dt_classifier(df, X_cols, y_cols):
    print("X cols:",X_cols)
    df = df[X_cols + y_cols].dropna()
    print(df.shape)
    X = df[X_cols]
    y = df[y_cols]
    print(f'df shape {df.shape}, X {X.shape}, y {y.shape}, {y_cols}')

    # Assuming X and y are already defined
    # Step 1: Split off the held-out validation set (15%)
    X_train_test, X_val, y_train_test, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # Step 2: Split the remaining 85% into training (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.2, random_state=42)

    # Print dataset shapes
    print('Check for set overlap:',set(X_train.index) & set(X_test.index), set(X_val.index) & set(X_test.index), set(X_train.index) & set(X_val.index))  # Should return an empty set
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Validation set size: {X_val.shape}")

    # Decision tree
    classifier = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=15, random_state=42, class_weight='balanced')

    # Cross validations
    cv = KFold(n_splits=5, shuffle=True, random_state=40)
    cross_val_results = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

    # Print CV results
    print(f"Cross-validation MSE scores: {-cross_val_results}")
    print(f"Average Cross-validation MSE scores: {-cross_val_results.mean()}")
    # Train and fit the model
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'classifier mean_absolute_error: {mae:.2f}, r2: {r2:.2f}')


    # # Test AUC-ROC
    y_pred_prob = classifier.predict_proba(X_test)[:, 1]  # Get probability of class 1
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f'\nAUC-ROC Score: {auc:.2f}, y_pred_prob = {y_pred_prob.mean():.2f}')

    # Find optimal parameters
    optimized_c = DecisionTreeClassifier(max_depth=2, min_samples_split=15, criterion='entropy',class_weight='balanced')
    optimized_c.fit(X_train, y_train)

    # Try optimized classifier on held-out validation set
    y_pred_val = optimized_c.predict(X_val)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nClassifier Held-out accuracy_score: {accuracy:.2f}')
    print(f'Classifier Held-out mean_absolute_error: {mae:.2f}, r2: {r2:.2f}')

    # Plot figure
    plt.figure(figsize=(12, 8))
    tree.plot_tree(optimized_c, feature_names=X.columns, filled=True)
    plt.show()

    return accuracy, mae, r2



# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX, verbose=False):
	# transform list into array (the row index and column names are lost)
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=500, n_jobs=-1)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(testX)
	return yhat[0]



# walk-forward validation for univariate data
def walk_forward_validation(data, test_prc, verbose=False):
	# Note: test_prc should be a value between 0 and 1, for example 0.2 for 20% split train/test data
	# Assuming you've already lesioned off your valiation data
	# Data should be in columns, with values in rows. 
	# X-data should be ALL BUT LAST COLUMN
	# y-data is LAST COLUMN
	rf_predictions = list()
	mean_predictions = list()
	# split dataset into TRAIN and TEST
	rows = data.shape[0]
	
	n_test = int(rows * test_prc)
	if isinstance(data, pd.DataFrame):
		data = np.asarray(data)
	test, train = train_test_split(data, test_size=test_prc) # Split based on test_prc
	if verbose:
		print('test shape', test.shape, 'train shape', train.shape)
	history = [row for row in train] 	# Retain rows of the NumPy array, turning into a list
	
	if len(history) > 0:

		# step over each time-step in the test set
		for i in range(len(test)):
			# Split the X and y-- assuming y = last column and X = all else
			testX = test[i, :-1]  # All columns except the last (testX)
			testy = test[i, -1]   # Last column (testy)			
			# Ensure testX is a 1D array or list (if needed)
			testX = np.asarray([testX])  
			testy = np.asarray([testy])  
			if verbose: 
				print(f'testX df is {testX.shape}; testy df is {testy.shape}')
			
			# fit model on history and make a prediction
			yhat = random_forest_forecast(history, testX, verbose)
			# store forecast in list of predictions
			rf_predictions.append(yhat)
			# add actual observation to history for the next loop
			history.append(test[i])

			# summarize progress
			if verbose:
				print('>expected=%.1f, predicted=%.1f' % (testy.item() if isinstance(testy, np.ndarray) else testy, yhat.item() if isinstance(yhat, np.ndarray) else yhat))	
		
	# After all predictions have been calculated, estimate prediction error
	error = mean_absolute_error(test[:, -1], rf_predictions)
	# Evaluate performance (e.g., using mean squared error)
	mse = mean_squared_error(test[:, -1], rf_predictions)
	# Evaluate R-squared
	r2 = sklearn.metrics.r2_score(test[:, -1],rf_predictions)

	# Return the error, the real data and the predicted data
	return error, test[:, -1], rf_predictions, mse, r2


