import numpy as np
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.decomposition import PCA
from collections import Counter
from source import util

def pca(X, numComponents):
    pca = PCA(n_components=numComponents)
    pca.fit(X)
    PCA(copy=True, iterated_power='auto', n_components=numComponents, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
    
    return pca.transform(X)
       
    
def kMeans(X, k):
    return KMeans(n_clusters=k).fit(X)


def svmClassifier(X, y):
    clf = svm.SVC()
    clf.fit(X, y)
    
    svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    
    return clf
    

def gmm(points, n_classes):
    clf = mixture.GaussianMixture(n_components=n_classes, covariance_type='full')
    pdfs = np.exp(clf.fit(points).score_samples(points))
        
    return pdfs


def kde(points, n_classes):
    kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(points)
    pdfs = np.exp(kernel.score_samples(points))
    
    return pdfs


def majorityVote(clusteredData, clusters, y):
    kPredicted = []
    
    for i in range(len(clusteredData)):
        group = clusteredData[i]
        #print("Grupo: ", group)
        ind = np.where(clusters==group)
        #print("Indices: ", ind)
        label = y[ind]
        #print("labels: ",label)
        voting = Counter(label).most_common(1)[0][0]
        #print(voting)
        kPredicted.append(voting)
    
    return kPredicted


def clusterAndLabel(X, y, ut, K):
    arrPredicted=[-1]*len(ut)
    
    for k in range(2, K+2):
        kmeans = kMeans(pca(X, 2), k)
        clusters = kmeans.labels_
        clusteredData = util.baseClassifier(pca(ut, 2), kmeans)
        arrPredicted=np.vstack([arrPredicted, majorityVote(clusteredData, clusters, y)])
    
    labels=[]
    for j in range(len(ut)):
        labels.append(Counter(arrPredicted[:, j]).most_common(1)[0][0])
    
    return labels