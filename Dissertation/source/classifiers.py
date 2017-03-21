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
    #clf = svm.LinearSVC()
    clf = svm.SVC()
    clf.fit(X, y)
    
    svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    '''
    svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
    '''
    return clf
    

def gmmWithBIC(X):
    ctype = 'full' #'spherical', 'tied', 'diag', 'full'
    best_gmm = False
    lowest_bic = np.infty
    bic = []
    n_components_range = [1, 2, 3, 5, 10, 15, 20]
    lenPoints = len(X)
    numComponentsChosen = 0

    for numComponents in n_components_range:
        if lenPoints >= numComponents:
            GMM = mixture.GaussianMixture(n_components=numComponents, covariance_type=ctype)
            GMM.fit(X)
            bic.append(GMM.bic(X))

            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = GMM
                numComponentsChosen = numComponents
    #return pdfs of best GMM model
    if best_gmm != False:
        #print("Best number of components: ",numComponentsChosen)
        return best_gmm
    else:
        print("gmmWithBIC: Error to choose the best GMM model")


def gmm(points, numComponents):
    clf = mixture.GaussianMixture(n_components=numComponents, covariance_type='full')
    clf.fit(points)
    return clf
    #pdfs = np.exp(clf.fit(allPoints).score_samples(points))   
    #return pdfs
    

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


def clusterAndLabel(X, y, Ut, K):
    arrPredicted=[-1]*len(Ut)
    lenPoints = len(X)
    
    for k in range(2, K+2):
        if lenPoints >= k:
            kmeans = kMeans(X, k)
            clusters = kmeans.labels_
            clusteredData = util.baseClassifier(Ut, kmeans)
            arrPredicted=np.vstack([arrPredicted, majorityVote(clusteredData, clusters, y)])
    
    labels=[]
    for j in range(len(Ut)):
        labels.append(Counter(arrPredicted[:, j]).most_common(1)[0][0])
    
    return labels