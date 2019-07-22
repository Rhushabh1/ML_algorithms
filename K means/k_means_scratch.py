import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np 

X = np.array([[1, 2],
			  [1.5, 1.8],
			  [5, 8],
			  [8, 8],
			  [1, 0.6],
			  [9, 11]])

# plt.scatter(X[:,0], X[:,1], c='b',s=5, linewidths=5)
# plt.show()

colors = 10*['g', 'r', 'c', 'b', 'k']


class K_Means:
	def __init__(self, k=2, tol=0.0001, max_iter=300):
		self.k = k
		self.max_iter = max_iter
		self.tol = tol

	def fit(self, data):
		
		self.centroids = {}
		
		for i in range(self.k):
			# first k centroids are gonna be first k points of data
			self.centroids[i] = data[i]

		for i in range(self.max_iter):
			# key is class number, data is featureset points in that class
			self.classifications = {}

			for i in range(self.k):
				# initializing all classes as []
				self.classifications[i] = []

			for featureset in data:
				# picking up the centroid closest to the featureset
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids = dict(self.centroids)

			for classification in self.classifications:
				# the shifting of centroids to the new centroid of classes
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)

			optimized = True

			for c in self.centroids:
				# for comparing the previous and new centroids for tolerance value
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid-original_centroid)/original_centroid*100.0) >= self.tol:
					optimized = False

			if optimized:
				break

	def predict(self, data):
		distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

clf = K_Means()
clf.fit(X)

# for plotting all the centroids
for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
		        marker = 'o', color = 'b', s= 5, linewidths = 5)

# for plotting all featureset points related to all the classifications
for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker='x', c= color, s= 20, linewidths=5)

unknowns = np.array([[1,3],
					 [8,9],
					 [0,3],
					 [5,4],
					 [6,4]])

for unknown in unknowns:
	classification = clf.predict(unknown)
	plt.scatter(unknown[0], unknown[1], marker = "*", s=5, color = colors[classification], linewidths = 5)


plt.show()
