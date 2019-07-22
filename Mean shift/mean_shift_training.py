import numpy as np 
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('ggplot')

centers = [[1,1,1], [3,10,10], [5,5,5]]
# this is used to generate datasets from just the centers of the blobs
X, _ = make_blobs(n_samples = 100, centers = centers, cluster_std = 1)

ms = MeanShift()
ms.fit(X)

# y 
labels = ms.labels_ # this has all the classification of all the featureset points
cluster_centers = ms.cluster_centers_ # returns the centroids
print(cluster_centers)

n_clusters_ = len(np.unique(labels)) # for unique classifications
print('number of estimated clusters: ', n_clusters_)
# accuracy = np.sum((centers-cluster_centers)/centers*100)
# print('accuracy: ', accuracy)

# data is done, now the plotting part

colors = 10*['g','r','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
	ax.scatter(X[i][0], X[i][1], X[i][2], c = colors[labels[i]], marker = 'o')

ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2],
		   marker = 'x', color = 'k', s = 50, linewidths = 5, zorder = 10)

plt.show()
