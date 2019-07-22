import pandas as pd
import quandl 
# for importing data of stock market price values
import math
import numpy as np
# It helps to deal with arrays
from sklearn import preprocessing, svm
# feature scaling and mean normalization , svm uses kernels
from sklearn.model_selection import *
# split data to test and get an unbiased classifier (cross_validate), train_test_split() 
from sklearn.linear_model import LinearRegression
# the main linear regression module

# we import data with this line
df = quandl.get('WIKI/GOOGL') # df - dataframe 

#  to print the data of the dataframe df
# print(df.head())

# which useful columns to pick up
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
# new columns added -- high-low percent , percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] *100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] *100 

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

# column to be predicted
forecast_col = 'Adj. Close'
# fill up the column with -99999 where there is empty space
df.fillna(-99999, inplace = True)

# how many values ahead of the current value to forecast
forecast_out  = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
# drops the missing data
df.dropna(inplace=True)

# it involves stock market strategy and calculation
# we have defined our features and labels up till this point

x = np.array(df.drop(['label'],1)) # features that are used to predict y
y = np.array(df['label'])

x = preprocessing.scale(x) # don't forget to scale the new values too alongside the old values that you used to train the classifier
y = np.array(df['label'])

# x = x[:-forecast_out+1]

print(len(x) , len(y))

# split the data to get other useful arrays
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

# we can use LinearRegression directly or use SVM
# n_jobs indicates how many processes it can run in parallel
classifier = LinearRegression(n_jobs = -1)

# kernel-type can be specified in svm 

# SVR = support vector regression
# classifier = svm.SVR(kernel='poly') # very low accuracy , almost equvalent to coin flip (50 percent)
# classifier = svm.SVR()

# we fit the data in our classifier (training)
classifier.fit(x_train, y_train)
# we take the score from the testing data and this indicates accuracy (testing)
accuracy = classifier.score(x_test, y_test)

print(accuracy)










