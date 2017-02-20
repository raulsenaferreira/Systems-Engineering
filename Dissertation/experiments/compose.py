import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans

'''
Eight  features  (temperature,  dew  point,
sea-level  pressure,  visibility,  average  wind  speed,  max  sus-
tained wind speed, and minimum and maximum temperature)
are  used  to  determine  whether  each  day  experienced  rain  or
no rain.
'''
data = pd.read_csv('C:\\Users\\user\\Documents\\Dissertacao\\gsod_2017\\007026-99999-2017.op',sep = "\t")
print(data)
n_samples = 30#300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

#Test
randd = np.random.randn(10, 3)

# concatenate the two datasets into the final training set
#X_train = np.vstack([shifted_gaussian, stretched_gaussian])
X_train = np.vstack(randd)

#K-means
kmeans = KMeans(n_clusters=2).fit(X_train)
clusters = kmeans.labels_

#Slicing instances according to their inferred clusters
label1 = 0
label2 = 1
trainClasses = {}
trainClasses[label1]=[X_train[i] for i in range(0,len(clusters)) if clusters[i] == label1]
trainClasses[label2]=[X_train[i] for i in range(0,len(clusters)) if clusters[i] == label2]

# fit a Gaussian Mixture Model with two components
gmm = {}
for c, points in trainClasses.items():
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clf.fit(points)
    gmm[c] = clf.score_samples(points)

#Kernel density estimation
kde={}
for c, points in trainClasses.items():
    kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(points)
    kde[c] = kernel.score_samples(points)

#print("Original data: ",X_train, "\n")
#print("K-Means clusters: ", clusters, "\n")
#print("Points from Cluster", label1, ": ", trainClasses[label1], "\n")
#print("Points from Cluster", label2, ": ", trainClasses[label2], "\n")
#print("PDF for class", label1, "using GMM: ", gmm[label1], "\n")
#print("PDF for class", label2, "using GMM: ", gmm[label2], "\n")
#print("PDF for class", label1, "using KDE: ", kde[label1], "\n")
#print("PDF for class", label2, "using KDE: ", kde[label2], "\n")
