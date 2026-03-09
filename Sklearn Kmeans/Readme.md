# K-Means Clustering – Customer Segmentation
# Overview

This project demonstrates the use of the K-Means Clustering algorithm to group customers based on their Age and Income.

Clustering helps in identifying patterns in data without using labeled outputs. In this project, customers are divided into different clusters depending on their similarities.

# Dataset

The dataset contains the following attributes:

Column	Description
Name	Customer name
Age	Age of the customer
Income ($)	Annual income of the customer

Example:

Name	Age	Income ($)
Rob	27	70000
Michael	29	90000
Mohan	29	61000
Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-learn

Steps Performed
1. Data Loading

The dataset is loaded using Pandas DataFrame.

2. Data Preprocessing

Removed non-numeric column Name for clustering

Scaled features using MinMaxScaler

3. Applying K-Means Clustering

The K-Means algorithm was used to divide the dataset into 3 clusters.

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3)
y_pred = km.fit_predict(X)
4. Finding Optimal K (Elbow Method)

The Elbow Method is used to determine the best number of clusters.

sse = []

for k in range(1,10):
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)
5. Visualization

Clusters are visualized using Matplotlib scatter plot.

plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='blue', label='Cluster 2')

plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            color='purple',
            marker='+',
            s=200,
            label='Centroids')
Output

The algorithm groups customers into three clusters based on their income and age.
Centroids represent the center of each cluster.

# Project Structure
KMeans-Clustering/
│
├── dataset.csv
├── kmeans_clustering.ipynb
├── clustering_visualization.png
└── README.md
Key Concepts Used

# Unsupervised Learning

<img width="792" height="568" alt="image" src="https://github.com/user-attachments/assets/76921ec5-b540-4cb3-b1c6-e9d6c6faacd3" />

# K-Means Clustering

<img width="731" height="535" alt="image" src="https://github.com/user-attachments/assets/bc8d4f0c-6e49-42b2-8211-0bf830f0f579" />

# Centroids

<img width="712" height="550" alt="image" src="https://github.com/user-attachments/assets/c6bc96e7-d491-4448-a227-c234d21d5e03" />

# MinMax Scaling

<img width="718" height="531" alt="image" src="https://github.com/user-attachments/assets/7d8809e8-4308-4329-9074-0a898fe75ecc" />

# Elbow Method

<img width="700" height="530" alt="image" src="https://github.com/user-attachments/assets/59567209-61ca-4089-8d08-bbac943e114b" />

# Results

The clustering model successfully groups customers based on similar income and age patterns, which can help in:

Customer segmentation

Targeted marketing

Business analytics

Author

Parmeet
Student – Data Science & Machine Learning


