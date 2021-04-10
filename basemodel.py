import pandas as pd
import numpy as np
import csv

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def ModelPredict(X, Y):
	"""
	This function predicts phenotype values using Lasso Regression
	"""
	RUNS = 20
	FEATURES = X.shape[1]
	ALPHA = 0.01
	w_list = []
	weight_runs = np.zeros((FEATURES, RUNS))
	#determine weights that appear over a period of runs
	for run in range(RUNS):
	    print("Calculating weights... Run", run+1 , "of", RUNS)
	    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = True)
	    mod = Lasso(alpha = ALPHA, max_iter = 60000)
	    mod.fit(X_train, y_train)
	    betas = mod.coef_
	    for feature in range(FEATURES):
	        if betas[feature] != 0:
	            weight_runs[feature, run] = 1
	print("Determining features that appear in more than half the runs...")
	for feature in range(FEATURES):
		sum_features = 0
		for run in range(RUNS):
			sum_features += weight_runs[feature,run]
		if sum_features > RUNS//2:	# if feature appears in more than half the runs, then it will be added to the list of weights
			w_list.append(feature)

    #w_list = [17, 47, 71, 237, 416, 447, 474, 526, 534, 693, 703, 755, 811, 813, 906, 921, 963, 967, 1081, 1229, 1303, 1319, 1322, 1324, 1325, 1377, 1387, 1421, 1509, 1516, 1582, 1585, 1611, 1612, 1670, 1697, 1732, 1733, 1800, 1959, 1972, 2013, 2026, 2047, 2133, 2134, 2164, 2172, 2206, 2230, 2244, 2251, 2290, 2294, 2331, 2339, 2340, 2366, 2509, 2610, 2629, 2630, 2657, 2664, 2672, 2673, 2686, 2722, 2801, 2816, 2888, 2967, 2970, 3021, 3039, 3073, 3236, 3252, 3267, 3279, 3283, 3299, 3374, 3510, 3624, 3642, 3878, 3890, 3925, 3937, 4079, 4187, 4247, 4278, 4410, 4418, 4436, 4668, 4672, 4725, 4733, 4754, 4760, 4783, 4797, 4814, 4821, 4822, 4945, 5043, 5060, 5071, 5074, 5084, 5106, 5114, 5182, 5360, 5691, 5712, 5725, 5742, 5747, 5790, 5896, 5900, 5954, 5958, 5968, 5987, 6003, 6154, 6158, 6180, 6181, 6270, 6536, 6740, 6774, 6957, 7028, 7113, 7202, 7210, 7213, 7224, 7234, 7237, 7248, 7257, 7283, 7307, 7388, 7427, 7461, 7462, 7473, 7474, 7491, 7492, 7495, 7514, 7527, 7763, 7795, 7798, 7986, 8020, 8063, 8072, 8122, 8208, 8285, 8289, 8328, 8535, 8553, 8573, 8605, 8643, 8661, 8663, 8782, 8834, 8927, 9204, 9362, 9363, 9366, 9369, 9536, 9703, 9722, 9859, 9921, 9946, 9960, 9971, 10055, 10096, 10318, 10421, 10444, 10481, 10504, 10524, 10563, 10647, 10656, 10696, 10701, 10702, 10799, 10816, 10858, 10871, 10919, 10938, 10960, 11135, 11138, 11161, 11168, 11181, 11186, 11188, 11191, 11203, 11217, 11279]
	#a
	print("Doing Lasso Regression...")
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = True)
	regmod = Lasso(alpha = ALPHA,max_iter = 60000)
	regmod.fit(X_train, y_train)
	for i in range(len(regmod.coef_)): #remove extraneous weights
		if i not in w_list:
			regmod.coef_[i] = 0
	
	print("Predicting...")
	pred = regmod.predict(X_test)
	print("R^2 Score:", r2_score(y_test, pred))
	print("MSE Score:", mean_squared_error(y_test, pred))
	print("MAE Score:", mean_absolute_error(y_test, pred))
	return pred
	
	