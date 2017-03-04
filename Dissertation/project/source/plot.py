{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "def plotDistributions(distributions):\n",
    "    i=0\n",
    "    #ploting\n",
    "    fig = plt.figure()\n",
    "    handles = []\n",
    "    colors = ['magenta', 'cyan']\n",
    "    classes = ['cluster 1', 'cluster 2']\n",
    "    ax = fig.add_subplot(121)\n",
    "    \n",
    "    for X in distributions:\n",
    "        #reducing to 2-dimensional data\n",
    "        x=pca(X, 2)\n",
    "        \n",
    "        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))\n",
    "        i+=1\n",
    "    \n",
    "    ax.legend(handles, classes)\n",
    "    \n",
    "    plt.show()\n",
    "    \n",
    "    \n",
    "def plotDistributionByClass(instances, indexesByClass):\n",
    "    i=0\n",
    "    #ploting\n",
    "    fig = plt.figure()\n",
    "    handles = []\n",
    "    colors = ['magenta', 'cyan']\n",
    "    classes = ['cluster 1', 'cluster 2']\n",
    "    ax = fig.add_subplot(121)\n",
    "    \n",
    "    for c, indexes in indexesByClass.items():\n",
    "        X = instances[indexes]\n",
    "        #reducing to 2-dimensional data\n",
    "        x=pca(X, 2)\n",
    "        \n",
    "        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))\n",
    "        i+=1\n",
    "    \n",
    "    ax.legend(handles, classes)\n",
    "    \n",
    "    plt.show()\n",
    "    \n",
    "    \n",
    "def plotAccuracy(arr, label):\n",
    "    c = range(len(arr))\n",
    "    fig = plt.figure()\n",
    "    fig.add_subplot(122)\n",
    "    ax = plt.axes()\n",
    "    \n",
    "    ax.legend(ax.plot(c, arr, 'k'), label)\n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
