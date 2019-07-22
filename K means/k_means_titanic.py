import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Pclass (1, 2, 3)
# survival (0 = No, 1 = Yes)
# name
# sex
# age
# sibsp - number of siblings/spouses Aboard
# parch - number of parents/children Aboard
# ticket - ticket number 
# fare - each passenger paid
# cabin
# embarked
# boat - lifeboat
# body - id number
# home.dest

df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
# print(df.head())

def handle_non_numerical_data(df):
	columns = df.columns.values

	for column in columns:
		# the list that has all the keys like {"Female":0, "Male":1}
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1

			df[column] = list(map(convert_to_int, df[column]))

	return df

df = handle_non_numerical_data(df)
# print(df.head())

# df.drop(['ticket'], 1, inplace=True) # ticket, boat have very low dependency

# astype(float) converts all the datatypes to float
X = np.array(df.drop(['survived'], 1).astype(float))
# scaling values helped a lot in increasing the accuracy
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters = 2)
clf.fit(X)

# now we jsut have to check k-means by cross-verifying with y
correct = 0
for i in range(len(X)):
	predict_me = np.array(X[i].astype(float))
	# jsut reshaping it to be compatible with np
	predict_me = predict_me.reshape(1, len(predict_me))
	prediction = clf.predict(predict_me)
	if prediction[0] == y[i]:
		correct += 1

# the value of accuracy can be vague, because k-means classifies the first cluster it sees as 0
# so you can always flip the value to 1-accuracy to get a higher value
print(correct/len(X))