import numpy as np
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.decomposition import PCA
from collections import Counter
from source import util
#import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture

def pca(X, numComponents):
    pca = PCA(n_components=numComponents)
    pca.fit(X)
    PCA(copy=True, iterated_power='auto', n_components=numComponents, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
    
    return pca.transform(X)
       
    
def kMeans(X, k):
    return KMeans(n_clusters=k).fit(X)


def svmClassifier(X, y):
    #clf=svm.SVC()
    '''
    if isImbalanced:
        clf = svm.SVC(C=1.0, cache_size=200, kernel='linear', class_weight='balanced', coef0=0.0,
            decision_function_shape=None, degree=3, gamma='auto', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        return clf.fit(X, y)
    else:'''
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=4, gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    #clf = svm.SVC(gamma=2, C=1)
    return clf.fit(X, y)
    

def randomForest(X, y):
    num_trees = 100
    max_features = np.ndim(X)
    return RandomForestClassifier(n_estimators=num_trees, max_features=max_features).fit(X, y)
    

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


def gmmWithPDF(points, allPoints, numComponents):
    if len(allPoints) < numComponents:
        numComponents=len(allPoints)
    clf = mixture.GaussianMixture(n_components=numComponents, covariance_type='full')
    clf.fit(allPoints)
    return np.exp(clf.score_samples(points))   
    

def bayesianGMM(points, allPoints, numComponents):
    if len(allPoints) < numComponents:
        numComponents=len(allPoints)
    clf = BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=100000,
        n_components=2 * numComponents, reg_covar=0, init_params='random',
        max_iter=1500, mean_precision_prior=.8,
        random_state=2)
    clf.fit(allPoints)
    return np.exp(clf.score_samples(points))


def kde(points, n_classes):
    kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(points)
    pdfs = np.exp(kernel.score_samples(points))
    
    return pdfs


def majorityVote(clusteredData, clusters, y):
    kPredicted = []
    
    for i in range(len(clusteredData)):
        group = clusteredData[i]
        ind = np.where(clusters==group)[0]
        #label = np.squeeze(np.asarray(y[ind]))
        label=y[ind]
        voting = Counter(label).most_common(1)[0][0]
        kPredicted.append(voting)
    
    return kPredicted


def clusterAndLabel(X, y, Ut, K, classes):
    labels=[]
    initK = len(classes)
    #arrPredicted=np.array([-1]*len(Ut))
    arrPredicted = []
    lenPoints = len(X)
    #print(len(Ut))
    #print(lenPoints)
    for k in range(initK, K+initK):
        if lenPoints >= k:
            #print("lllll")
            kmeans = kMeans(X, k)
            clusters = kmeans.labels_
            clusteredData = kmeans.predict(Ut)
            #arrPredicted=np.vstack([arrPredicted, majorityVote(clusteredData, clusters, y)])
            arrPredicted.append(majorityVote(clusteredData, clusters, y))
    
    arrPredicted = np.array(arrPredicted)
    #print(arrPredicted)
    #print(len(Ut))
    for j in range(len(Ut)):
        #print(arrPredicted[:, j])
        labels.append(Counter(arrPredicted[:, j]).most_common(1)[0][0])
    #print(labels)
    return labels


def classify(X, y, Ut, K, classes, clf='rf'):
        if clf=='svm':
            #print("Using SVM")
            clf = svmClassifier(X, y)
            return clf.predict(Ut)
        elif clf=='cl':
            #print("Using cluster and label")
            return clusterAndLabel(X, y, Ut, K, classes)
        elif clf=='rf':
            #print("Using Random Forest")
            clf = randomForest(X, y)
            return clf.predict(Ut)
