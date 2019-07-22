import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
from sklearn.cluster import MeanShift
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
original_df = pd.DataFrame.copy(df)

df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
	columns = df.columns.values

	for column in columns:
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

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

# we will save to which cluster that passenger belongs to in df
original_df['cluster_group'] = np.nan 

# iterate throught the labels and copy in new column
for i in range(len(X)):
	# iloc[i] returns value of 'i'th row
	# the ith row of the column 'cluster_group' = labels[i]
	original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

# its possible that there are different number of groups instead of just 2
survival_rates = {}
for i in range(n_clusters_):
	# temporary df which has all the passengers in cluster i
	temp_df = original_df[ (original_df['cluster_group'] == float(i)) ]
	# the ones who survived among them
	survival_cluster = temp_df[ (temp_df['survived'] == 1) ]
	# the fraction survived
	survival_rate = len(survival_cluster)/ len(temp_df)
	survival_rates[i] = survival_rate

print('###################################################')
print(survival_rates)
print('###################################################')
# describe() prints a compact and concise data of the df where passengers belong to cluster 0
print(original_df[ (original_df['cluster_group'] == 0) ].describe())
print('###################################################')
print(original_df[ (original_df['cluster_group'] == 1) ].describe())
print('###################################################')
print(original_df[ (original_df['cluster_group'] == 2) ].describe())
print('###################################################')
print(original_df[ (original_df['cluster_group'] == 3) ].describe()) 
print('###################################################')

# passengers in cluster 0
cluster_0 = original_df[ (original_df['cluster_group']==0) ]
# the ones which are first-class-passengers
cluster_0_fc = cluster_0[ (cluster_0['pclass']==1) ]
print(cluster_0_fc.describe())