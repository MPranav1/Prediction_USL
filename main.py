import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('Iris.csv')

# Select the features to use for clustering
X = df[['SepalLengthCm', 'SepalWidthCm']]

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Perform K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)

# Add the cluster assignments to the DataFrame
df['cluster'] = kmeans.labels_

# Visualize the clusters
colors = ['red', 'green', 'blue', ...]  # add more colors as needed
for i in range(optimal_num_clusters):
    plt.scatter(df[df['cluster'] == i]['feature_1'], df[df['cluster'] == i]['feature_2'], color=colors[i])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
