import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans
from sklearn import svm
'''
Eight  features  (temperature,  dew  point,
sea-level  pressure,  visibility,  average  wind  speed,  max  sus-
tained wind speed, and minimum and maximum temperature)
are  used  to  determine  whether  each  day  experienced  rain  or
no rain.
'''
n_samples = 30#300
n_dimensions = 3
# generate random sample
np.random.seed(0)

#Test 1 with single distribution
randd = np.random.randn(n_samples, n_dimensions)

'''
#Test 2 with two different distributions
# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# concatenate the two datasets into the final training set
#X_train = np.vstack([shifted_gaussian, stretched_gaussian])
'''

X_train = randd

label1 = 0
label2 = 1
classes=[label1, label2]
numClasses = len(classes)


#K-means
kmeans = KMeans(n_clusters=numClasses).fit(X_train)
clusters = kmeans.labels_
#predictedClusters = kmeans.predict([[0, 0, 0], [4, 4, 5]]) #predciting

#Slicing instances according to their inferred clusters
trainClasses = {}
for c in range(numClasses):
    trainClasses[classes[c]]=[X_train[i] for i in range(len(clusters)) if clusters[i] == c]
    

# fit a Gaussian Mixture Model with n components
gmm = {}
for c, points in trainClasses.items():
    clf = mixture.GaussianMixture(n_components=6, covariance_type='full')
    clf.fit(points)
    gmm[c] = clf.score_samples(points)

#Kernel density estimation
kde={}
for c, points in trainClasses.items():
    kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(points)
    kde[c] = kernel.score_samples(points)

print("Original data: ",X_train, "\n")
print("K-Means clusters: ", clusters, "\n")
print("Points from Cluster", label1, ": ", trainClasses[label1], "\n")
print("Points from Cluster", label2, ": ", trainClasses[label2], "\n")
print("PDF for class", label1, "using GMM: ", gmm[label1], "\n")
print("PDF for class", label2, "using GMM: ", gmm[label2], "\n")
print("PDF for class", label1, "using KDE: ", kde[label1], "\n")
print("PDF for class", label2, "using KDE: ", kde[label2], "\n")

#Comparing the results of entire method above with SVM
X = X_train
y = kmeans.labels_ #putting kmeans labels as a response from specialist (yes, i'm lazy)
clf = svm.SVC()
clf.fit(X, y)
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)


print("Sample predicted by SVM: ", clf.predict([[0, 0, 0], [4, 4, 5]]))
