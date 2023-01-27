# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# Read in the data
df = pd.read_csv('credit_data_risk.csv')

# Data cleaning
df = df.dropna()

# Data overview
del df['Unnamed: 0']
print(df.describe())

numerical_credit =df.select_dtypes(exclude=['O'])

# Find clusters
scaler = StandardScaler()
scaled_credit = scaler.fit_transform(numerical_credit)

distance = []
for k in range(1, 10):
    km = KMeans(n_clusters=k)
    km.fit(scaled_credit)
    distance.append(km.inertia_)

# plt.plot(range(1, 10), distance, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.title('Elbow method')
# plt.show()

# Silhouette score
fig, ax = plt.subplots(4, 2, figsize=(10, 10))
plt.subplots_adjust(hspace=0.8)
for i in range(2,10 ):
    km = KMeans(n_clusters=i)
    q,r = divmod(i,2)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][r])
    visualizer.fit(scaled_credit)
    ax[q-1][r].set_title('Silhouette score for {} clusters'.format(i))
    ax[q-1][r].set_xlabel('Silhouette score')
    ax[q-1][r].set_ylabel('Cluster label')

plt.show()

kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(scaled_credit)

# Plot clusters
plt.scatter(scaled_credit[:, 0], scaled_credit[:, 1], c=clusters, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

