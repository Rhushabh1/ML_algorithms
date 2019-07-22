import pandas as pd
import quandl 
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
# serializing our classifier
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] *100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] *100 

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

# using forecast amount from the original data (returns fraction)
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
x = np.array(df.drop(['label'],1)) 
x = preprocessing.scale(x) 
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

print(len(x) , len(y))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

classifier = LinearRegression(n_jobs = -1)
classifier.fit(x_train, y_train)
# saving the classifier using pickle
with open('LinearRegression.pickle','wb') as file:
	pickle.dump(classifier, file)

pickle_in = open('LinearRegression.pickle', 'rb') 
classifier = pickle.load(pickle_in)

accuracy = classifier.score(x_test, y_test)
forecast_set = classifier.predict(x_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
