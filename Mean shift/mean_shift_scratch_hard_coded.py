import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np 

X = np.array([[1, 2],
			  [1.5, 1.8],
			  [5, 8],
			  [8, 8],
			  [1, 0.6],
			  [9, 11],
			  [8,2],
			  [10,2],
			  [9,3]])

# plt.scatter(X[:,0], X[:,1], c='b',s=5, linewidths=5)
# plt.show()

colors = 10*['g', 'r', 'c', 'b', 'k']

class Mean_Shift:
	# radius should be self-regulating according to data and not hard-coded
	def __init__(self, radius=4):
		self.radius = radius


	def fit(self, data):

		# we want centroids to be numbered
		centroids = {}

		# all data points are centroids at first
		for i in range(len(data)):
			# {0:pt0, 1:pt1}
			centroids[i] = data[i]


		while True:
			# list of tuple of new_centroid
			new_centroids = []

			for i in centroids:
				# has data points which are in the radius of the centroid
				in_radius = []
				centroid = centroids[i]

				for featureset in data:
					if np.linalg.norm(featureset - centroid) < self.radius:
						in_radius.append(featureset)


				# new_centroid will naturally be more closer to close data points
				new_centroid = np.average(in_radius, axis = 0)
				new_centroids.append(tuple(new_centroid))

			# sorted so that the order is intact for comparison and terminating-condition-step
			uniques = sorted(list(set(new_centroids)))

			# pop those centroids which are seemingly close to each other within the tolerance range (default = self.radius)
			to_pop = []

			prev_centroids = dict(centroids)

			# again it is NULL
			centroids = {}
			# copying uniques into controids
			for i in range(len(uniques)):
				# centroid = {0:pt0, 1:pt1}
				centroids[i] = np.array(uniques[i])

			optimized = True

			# terminating condition of the loop is convergence
			# prev_centroids = centroids
			for i in centroids:
				if not np.array_equal(centroids[i], prev_centroids[i]):
					optimized = False

				if not optimized:
					break

			if optimized:
				break

		self.centroids = centroids

		# to store the featuresets in each cluster together
		self.classifications = {}

		for i in range(len(self.centroids)):
			self.classifications[i] = []

		for featureset in data:
			# the prediction of the featureset to clusters
			distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
			classification = distances.index(min(distances))
			self.classifications[classification].append(featureset)

	def predict(self, data):
		distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification


clf = Mean_Shift()
clf.fit(X)

# cluster_centers
centroids = clf.centroids

# the plotting part
# plt.scatter(X[:,0], X[:,1], s=5, linewidths = 5)

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker ='x', color= color, s = 10,linewidths = 5)

for c in centroids:
	plt.scatter(centroids[c][0], centroids[c][1], color = 'k', marker='o', s=5, linewidths = 5)

plt.show()