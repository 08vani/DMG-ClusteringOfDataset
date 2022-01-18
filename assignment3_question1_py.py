# -*- coding: utf-8 -*-
"""Assignment3_Question1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZOiO5JmTN0CRV7XdXqASUoEbvvDnSTSz

#  **Question 1 Clustering modeling**
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsne
import seaborn as sns
import tensorflow as tf
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_mutual_info_score

"""Importing the csv file"""

data = pd.read_csv('/content/covtype_train.csv')
data.head()

data.drop(data.tail(402598).index, inplace = True)
data.target = data.target - 1
true_labels= data['target']
data.drop(columns=['target'], inplace=True)
data.head()
data.shape

"""To get the numerical labels"""

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()

data['Elevation']= label_encoder.fit_transform(data['Elevation']) 
data['Aspect']= label_encoder.fit_transform(data['Aspect']) 
data['Slope']= label_encoder.fit_transform(data['Slope']) 
data['Wilderness']= label_encoder.fit_transform(data['Wilderness']) 
data['Soil_Type']= label_encoder.fit_transform(data['Soil_Type']) 
data['Hillshade_9am']= label_encoder.fit_transform(data['Hillshade_9am']) 
data['Hillshade_Noon']= label_encoder.fit_transform(data['Hillshade_Noon']) 
data['Horizontal_Distance_To_Hydrology']= label_encoder.fit_transform(data['Horizontal_Distance_To_Hydrology']) 
data['Vertical_Distance_To_Hydrology']= label_encoder.fit_transform(data['Vertical_Distance_To_Hydrology']) 
data['Horizontal_Distance_To_Fire_Points']= label_encoder.fit_transform(data['Horizontal_Distance_To_Fire_Points'])

data.head(20)

dt = np.array(data)
print(data.shape)
print(dt)

data[:]

data = tsne(n_components=2).fit_transform(dt)

data

"""

```
# This is formatted as code
```

## **K-mean Clustering**"""

kmeans = KMeans(n_clusters=7,max_iter=60,random_state=100).fit(data)
kmean_labels = kmeans.labels_
print(kmean_labels)

"""Finding the centroids"""

kmean_centroids = kmeans.cluster_centers_
print(kmean_centroids)

"""Visualization of the clusters"""

#https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/

frame = pd.DataFrame(data)
frame['cluster'] = kmean_labels
frame.columns = ['Weight', 'Height', 'cluster']

color=['blue','green','cyan', 'black','red','purple','yellow']
for k in range(0,7):
    reduced_data = frame[frame["cluster"]==k]
    plt.scatter(reduced_data["Weight"],reduced_data["Height"],c=color[k])
plt.show()

label_list = list(np.unique(kmean_labels))
PtsInCluster = []
for val in label_list:
  PtsInCluster.append(list(kmean_labels).count(val))
print(PtsInCluster)

"""Comparison between cluster distribution and true label count."""

adjusted_mutual_info_score(true_labels, kmean_labels)

"""## **DBSCAN Clustering**"""

dbscan = DBSCAN(eps=16, min_samples=4).fit(data)
dbscan_labels = dbscan.labels_

"""Finding the centroids"""

from sklearn.neighbors import NearestCentroid
clf=NearestCentroid()
X=dbscan.fit_predict(data)
clf.fit(data,X)
print((clf.centroids_))

"""Visualization of the clusters"""

#https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/
frame = pd.DataFrame(data)
frame['cluster'] = dbscan_labels
frame.columns = ['Weight', 'Height', 'cluster']

color=['blue','green','cyan', 'black','red','purple','yellow']
for k in range(0,7):
    reduced_data = frame[frame["cluster"]==k]
    plt.scatter(reduced_data["Weight"],reduced_data["Height"],c=color[k])
plt.show()

label_list = list(np.unique(dbscan_labels))
PtsInCluster = []
for val in label_list:
  PtsInCluster.append(list(dbscan_labels).count(val))
print(PtsInCluster)

"""Comparison between cluster distribution and true label count."""

adjusted_mutual_info_score(true_labels, dbscan_labels)

"""## **Aglomerative Clustering**"""

from sklearn.cluster import AgglomerativeClustering

Agmc = AgglomerativeClustering(n_clusters=7).fit(data)
Agmc_labels = Agmc.labels_

"""Finding the centroids"""

from sklearn.neighbors import NearestCentroid
clf=NearestCentroid()
X=Agmc.fit_predict(data)
clf.fit(data,X)
print(clf.centroids_)

"""Visualization of the clusters"""

#https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/
frame = pd.DataFrame(data)
frame['cluster'] = Agmc_labels
frame.columns = ['Weight', 'Height', 'cluster']

color=['blue','green','cyan', 'black','red','purple','yellow']
for k in range(0,7):
    reduced_data = frame[frame["cluster"]==k]
    plt.scatter(reduced_data["Weight"],reduced_data["Height"],c=color[k])
plt.show()

label_list = list(np.unique(Agmc_labels))
PtsInCluster = []
for val in label_list:
  PtsInCluster.append(list(Agmc_labels).count(val))
print(PtsInCluster)

"""Comparison between cluster distribution and true label count."""

adjusted_mutual_info_score(true_labels, Agmc_labels)

"""## **Gaussian Clustering**"""

from sklearn.mixture import GaussianMixture
gsm = GaussianMixture(n_components=7)
gsm.fit(data)
gsm_labels = gsm.predict(data)

"""Finding the centroids"""

from sklearn.neighbors import NearestCentroid
clf=NearestCentroid()
X=gsm.fit_predict(data)
clf.fit(data,X)
print(clf.centroids_)

"""Visualization of the clusters"""

#https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/
frame = pd.DataFrame(data)
frame['cluster'] = gsm_labels
frame.columns = ['Weight', 'Height', 'cluster']

color=['blue','green','cyan', 'black','red','purple','yellow']
for k in range(0,7):
    reduced_data = frame[frame["cluster"]==k]
    plt.scatter(reduced_data["Weight"],reduced_data["Height"],c=color[k])
plt.show()

label_list = list(np.unique(gsm_labels))
PtsInCluster = []
for val in label_list:
  PtsInCluster.append(list(gsm_labels).count(val))
print(PtsInCluster)

"""Comparison between cluster distribution and true label count."""

adjusted_mutual_info_score(true_labels, gsm_labels)

"""# **Comaring Each mode with Gaussian Bases clustering model**"""

adjusted_mutual_info_score(gsm_labels, kmean_labels)

adjusted_mutual_info_score(gsm_labels, dbscan_labels)

adjusted_mutual_info_score(gsm_labels, Agmc_labels)