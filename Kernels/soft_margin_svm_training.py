import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)

# if we include id label, then we are almost at coin-flipping status
# because it is the most random thing in the dataset
df.drop(['id'],1,inplace=True)

x = np.array(df.drop(['class'],1)) # features
y = np.array(df['class']) # label (predicting class)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# SVC = support vector classifier
# it is a soft margin classifier, C is a parameter (slack coeff)

# 
clf = svm.SVC(C = 1.0, # coefficient of slack
	          kernel = 'rbf', # which type of kernel to be used
	          degree = 3, # degree of the polynomial function
	          gamma = 'auto', # kernel coeff for 'rbf','poly','sigmoid', if gamma is 'auto' then 1/n_features will be used instead
	          coef0 = 0.0, # significant only in 'poly' and 'sigmoid'
	          probability = False, # to enable probability estimates or not
	          shrinking = True, # to use shrinking heuristic
	          tol = 1e-3, # tolerance for stopping criteria
	          verbose = False, # it takes advantage of per-process runtime setting in libsvm, it may not work properly in a multi-threaded context
	          max_iter = -1, # hard limit on iterations within solver
	          decision_function_shape = None # 'ovo','ovr' or None
	          )
# there are a lot of attributes for clf

# support_ = indices of support vectors
# n_support_ = number of support vectors for each class
# intercept_ = constants in decision function

clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])

# it reshapes data according to what the classifier wants
# np.array(...).reshape({number of elements in the array}, {size of each element})
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
