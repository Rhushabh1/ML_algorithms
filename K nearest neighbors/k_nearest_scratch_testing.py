import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd 
import random

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups! idiot!')

	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	# print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k
	
	return vote_result, confidence

accuracies = []

for i in range(25):
	df = pd.read_csv('breast-cancer-wisconsin.data')
	df.replace('?',-99999,inplace=True)
	# ['id'] = to-drop-label, 1 = column is dropped
	df.drop(['id'],1,inplace=True)

	# to have a list of list to work on 
	# astype(float) = to ensure every entry is of type float and not some random string
	full_data = df.astype(float).values.tolist()
	random.shuffle(full_data)

	# my own version of train_test_split
	test_size = 0.2
	train_set = {2:[], 4:[]}
	test_set = {2:[], 4:[]}
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])

	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	# testing part
	correct = 0
	total = 0

	for group in test_set:
		for data in test_set[group]:
			vote, confidence = k_nearest_neighbors(train_set, data, k = 5)
			if vote == group:
				correct += 1
			# else:
			# 	print(confidence)
			total += 1

	accuracy = correct/total
	# print("Accuracy:", accuracy)
	accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))