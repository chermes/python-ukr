#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
UKR tests.

Author: Christoph Hermes
Created on November 17, 2014  22:50:23
"""

from time import time

import numpy as np
import numexpr as ne
import itertools
import matplotlib.pyplot as plt
## from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda, random_projection)
from scipy.optimize import leastsq

from ukr_core import ukr_bp, ukr_dX, ukr_E, ukr_project
import rprop


## digits = datasets.load_digits(n_class=6)
## Y = digits.data
## y = digits.target
## q = 2
## loo_cv = 2
## ## max_iter = 20000
## max_iter = 5000

iris = datasets.load_iris()
Y = iris.data
Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
y = iris.target
q = 2
loo_cv = 1
max_iter = 2000

## # noisy spiral dataset
## t = np.linspace(0, 6*np.pi ,300)
## k = .2
## x = t*np.cos(t) + k*(np.random.randn(t.shape[0])-0.5)
## y = t*np.sin(t) + k*(np.random.randn(t.shape[0])-0.5)
## Y = np.c_[x,y]
## q = 1
## loo_cv = 1
## max_iter = 5000

## # toy problem
## Y = np.vstack([(np.random.randn(10,2) * 0.1 + y) for y in [[-1, -1], [-1, +1], [+1, -1], [+1, +1]]])
## q = 1
## loo_cv = 1
## max_iter = 100

## # toy problem for matlab comparison
## Y = np.array([
    ## [1., 0.],
    ## [2., 0.],
    ## [3., 0.],
    ## ])
## X = np.array([
    ## [1.],
    ## [2.],
    ## [3.1],
    ## ])
## q = 1
## loo_cv = 1
## max_iter = 100

# kernels including their derivatives
## gaussian = (lambda x: np.exp(-.5 * x), lambda x: -.5 * np.exp(-.5 * x))
gaussian = (lambda x: ne.evaluate('exp(-.5 * x)'), lambda x: ne.evaluate('-.5 * exp(-.5 * x)'))
quartic = (lambda x: np.where(x<1, (1. - x)**2, np.zeros_like(x)), lambda x: np.where(x<1, -2. * (1. - x), np.zeros_like(x)))
## student = (lambda x: 1. / (1. + x), lambda x: -1. / ((1. + x)**2))
## student = (lambda x: ne.evaluate('1. / (1. + x)'), lambda x: ne.evaluate('-1. / ((1. + x)**2)'))
## student_n = (lambda x, n: (1. + x/n)**(-(n+1.)/2.), lambda x, n: -(n+1.)/2. * n**((n+1.)/2.) * (x+n)**(-(n+1.)/2.-1.) )
student_n = (lambda x, n: ne.evaluate('(1. + x/n)**(-(n+1.)/2.)'), lambda x, n: ne.evaluate('-(n+1.)/2. * n**((n+1.)/2.) * (x+n)**(-(n+1.)/2.-1.)') )
student_1 = (lambda x: student_n[0](x, 1), lambda x: student_n[1](x, 1))
student_2 = (lambda x: student_n[0](x, 2), lambda x: student_n[1](x, 2))
student_3 = (lambda x: student_n[0](x, 3), lambda x: student_n[1](x, 3))
student_9 = (lambda x: student_n[0](x, 9), lambda x: student_n[1](x, 9))

kernel = 'student_2'
#
if kernel == 'gaussian':
    k, k_der = gaussian
elif kernel == 'quartic':
    k, k_der = quartic
elif kernel == 'student_1':
    k, k_der = student_1
elif kernel == 'student_2':
    k, k_der = student_2
elif kernel == 'student_3':
    k, k_der = student_3
elif kernel == 'student_9':
    k, k_der = student_9

## import IPython; IPython.embed()
## import sys; sys.exit()
## B, P = ukr_bp(np.vstack((X,X)), X, k, k_der, loo_cv)

# LLE
X = None
B_ = None
embed_ = None
error = np.inf
embeddings = [decomposition.PCA(n_components=q)]
## for n in range(3, 25, 1):
    ## embeddings.append(manifold.locally_linear.LocallyLinearEmbedding(n, q, method='modified'))

for embeddingI, embedding in enumerate(embeddings):
    print 'Try embedding %2d/%2d: %s' % (embeddingI+1, len(embeddings), embedding.__class__.__name__)

    try:
        X_init_ = embedding.fit_transform(Y)
    except:
        continue

    # normalize initial hypothesis to X.T * X = I
    ## X_init_ = np.linalg.pinv(X_init_.T.dot(X_init_)).dot(X_init_.T).T
    X_init_ = (X_init_ - X_init_.mean(axis=0)) / X_init_.std(axis=0)

    # TODO: replace this by iRProp+?
    # optimze the scaling factor
    def residuals(p, Y_, X_):
        B, P = ukr_bp(X_ * p, k, k_der, loo_cv)
        E = (Y_ - ukr_project(Y, B)).flatten()
        return E
    p0 = np.ones((1,q))
    plsq = leastsq(residuals, p0, args=(Y, X_init_))
    X_init_ = X_init_ * plsq[0]

    # final projection error estimation
    B, P = ukr_bp(X_init_, k, k_der, loo_cv)
    err_ = ukr_E(Y, B)

    print ' Error: %f' % err_

    if err_ < error:
        error = err_
        X = X_init_
        B_ = B
        embed_ = embedding

# Summary:
print 'Using embedding', embed_.__class__.__name__
print ' Error: %f' % error

## import IPython; IPython.embed()
## import sys; sys.exit()

Y_orig_estim = ukr_project(Y, B_)
## plt.plot(Y[:,0], Y[:,1], 'b.')
## plt.plot(Y_orig_estim[:,0], Y_orig_estim[:,1], 'r.')
## plt.title('optimized initialization')
## plt.show()

## import IPython; IPython.embed()
## import sys; sys.exit()


iRpropPlus = rprop.iRpropPlus()

for iter in xrange(max_iter):
    print 'UKR iter %5d, Err=%9.6f' % (iter, iRpropPlus.E_prev)

    # derivative of X_model w.r.t. to the error gradient
    B, P = ukr_bp(X, k, k_der, loo_cv)
    dX = ukr_dX(X, Y, B, P)

    # reconstruction error
    E_cur = ukr_E(Y, B)

    X = iRpropPlus.update(X, dX, E_cur)

    ## # plot and save the results
    ## B, _ = ukr_bp(X, k, k_der, diagK=0, X=X)
    ## Y_estim = ukr_project(Y, B)
    ## if Y.shape[1] <= 2:
        ## plt.figure(1)
        ## plt.clf()
        ## if q > 1:
            ## plt.subplot(1,2,1)
        ## plt.plot(Y[:,0], Y[:,1], 'b.', label='gt')
        ## plt.plot(Y_orig_estim[:,0], Y_orig_estim[:,1], 'g.', label='before RPROP')
        ## plt.plot(Y_estim[:,0], Y_estim[:,1], 'r.', label='after RPROP')
        ## plt.legend(loc='best')
        ## if q > 1:
            ## plt.subplot(1,2,2)
            ## plt.plot(X[:,0], X[:,1], '.')
    ## else: # high-dimensional data
        ## plt.figure(1)
        ## plt.clf()
        ## clrs = itertools.cycle('rgbcykm')
        ## mrks = itertools.cycle('.x+')
        ## for yi in np.unique(y):
            ## plt.plot(X[y==yi,0], X[y==yi,1], clrs.next() + mrks.next(), label=str(yi))
        ## plt.legend(loc='best')
        ## plt.title(kernel)
    ## plt.savefig('iter_%06d.png' % iter)

## plt.figure(1)
## plt.clf()
## plt.subplot(1,2,1)
## plt.plot(Y_lle[:,0], Y_lle[:,1], '.')
## #
## plt.subplot(1,2,2)
## plt.plot(Y_lle_orig[:,0], Y_lle_orig[:,1], '.')
## plt.title('orig')
## plt.show()

# plot the results
B, _ = ukr_bp(X, k, k_der, diagK=0, X=X)
Y_estim = ukr_project(Y, B)
if Y.shape[1] <= 2:
    plt.figure(1)
    if q > 1:
        plt.subplot(1,2,1)
    plt.plot(Y[:,0], Y[:,1], 'b.', label='gt')
    plt.plot(Y_orig_estim[:,0], Y_orig_estim[:,1], 'g.', label='before RPROP')
    plt.plot(Y_estim[:,0], Y_estim[:,1], 'r.', label='after RPROP')
    plt.legend(loc='best')
    if q > 1:
        plt.subplot(1,2,2)
        plt.plot(X[:,0], X[:,1], '.')
else: # high-dimensional data
    plt.figure(1)

    clrs = itertools.cycle('rgbcykm')
    mrks = itertools.cycle('.x+')
    for yi in np.unique(y):
        plt.plot(X[y==yi,0], X[y==yi,1], clrs.next() + mrks.next(), label=str(yi))
    plt.legend(loc='best')

    plt.title(kernel)

plt.show()

import IPython; IPython.embed()
