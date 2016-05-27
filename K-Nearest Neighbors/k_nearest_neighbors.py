import numpy as np
import pandas as pd
import random
import warnings
from collections import Counter

def k_nearest_neighbors(data, predict, k=3):
	'''
	Args:
		data (dict): each key is a class label pointing to a list of features
		predict (list)
	'''
	if len(data) >= k:
		warnings.warn('K is set a value less than total voting groups!')
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / float(k)

	return vote_result, confidence

accuracies = []
num_trials = 5

for i in range(num_trials):
	# parse csv
	df = pd.read_csv('breast-cancer-wisconsin-data.txt')
	df.replace('?', -99999, inplace=True)
	df.drop(['id'], 1, inplace=True)

	# make sure every cell is a float
	full_data = df.astype(float).values.tolist()
	random.shuffle(full_data)

	# train-test split
	test_size = 0.2
	train_set = {2:[], 4:[]}
	test_set = {2:[], 4:[]}
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])
	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0.0
	total = 0.0

	for group in test_set:
		for data in test_set[group]:
			vote, confidence = k_nearest_neighbors(train_set, data, k=5)
			if group == vote:
				correct += 1
			total += 1

	# add the results to the list
	accuracies.append(correct / total)
print('Average accuracy:', sum(accuracies) / len(accuracies))
