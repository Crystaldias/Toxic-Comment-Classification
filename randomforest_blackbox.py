# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:29:40 2018

@author: Granger
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def make_mesh(X):
    h =  0.02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    return xx,yy

def run_mesh(xx,yy,clf):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return Z

def Decision_Regions(X,clf,X_train,y_train,X_test,y_test,score):
    xx,yy = make_mesh(X)
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    figure = plt.figure(figsize=(18,8))
    ax = plt.subplot(1, 2, 1)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    #Show the test accuracy (score)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    plt.tight_layout()
    plt.show()
    
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
np.random.seed(100)

X,y=make_moons(noise=0.05, random_state=1)

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)

Decision_Regions(X,clf,X_train,y_train,X_test,y_test,score)