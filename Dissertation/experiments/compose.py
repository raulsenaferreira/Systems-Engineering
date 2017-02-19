import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
import pandas as pd

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

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)
'''
# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
'''

#Kernel density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_train)
kde = -kde.score_samples(X_train)

gmm = -clf.score_samples(X_train)

print("Dados originais: ",X_train, "\n")
print("PDF gerado pelo GMM: ", gmm, "\n")
print("PDF gerado pelo KDE: ", kde)
