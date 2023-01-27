# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load data
data = load_iris()
df = data.data
df = df[:, 1:3]

# Create linkage matrix
agg_clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')

# Cluster labels
labels = agg_clustering.fit_predict(df)

## Plot
plt.figure(figsize=(8, 5))
plt.scatter(df[labels==0,0], df[labels==0,1], s=50, c='red', label='Cluster 1')
plt.scatter(df[labels==1,0], df[labels==1,1], s=50, c='blue', label='Cluster 2')
plt.scatter(df[labels==2,0], df[labels==2,1], s=50, c='green', label='Cluster 3')
plt.show()

# Plot dendrogram
z = linkage(df, method='average', metric='euclidean')
dendogram = dendrogram(z)
plt.title('Dendrogram')
plt.show()