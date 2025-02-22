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


############ Individual Random Forest Walk-forward Validation & TT Split Functions ##############
# split a univariate dataset into train/test sets
# based on a percentage of rows for test dataset by 'test_prc' (for example, 0.2)
def train_test_split(data, n_test, verbose=False):
	if verbose:
		print(f'train_test_split: Train rows = {data.shape[0] - n_test}; Test rows = {n_test}')
	
 # If data is already a NumPy array, use normal slicing
	return data[:n_test, :], data[n_test:, :]

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
	
	test, train = train_test_split(data, n_test, verbose) # Split based on test_prc
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