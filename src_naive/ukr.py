#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unsupervised Kernel Regression (UKR) for Python.
Implemented as a scikit-learn module.

Author: Christoph Hermes
Created on Januar 16, 2015  18:48:22


The MIT License (MIT)

Copyright (c) 2015 Christoph Hermes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import numpy as np
from scipy.optimize import leastsq
import sklearn
from sklearn import decomposition, manifold

# own modules
from ukr_core import (ukr_bp, ukr_dY, ukr_E, ukr_project,
        ukr_backproject_particles)
import rprop


# possible UKR kernels: tuple(kernel, kernel derivative)
try: # try using numexpr
    import numexpr as ne
    gaussian = (lambda x: ne.evaluate('exp(-.5 * x)'), lambda x: ne.evaluate('-.5 * exp(-.5 * x)'))
    quartic = (lambda x: np.where(x<1, (1. - x)**2, np.zeros_like(x)), lambda x: np.where(x<1, -2. * (1. - x), np.zeros_like(x)))
    student_n = (lambda x, n: ne.evaluate('(1. + x/n)**(-(n+1.)/2.)'), lambda x, n: ne.evaluate('-(n+1.)/2. * n**((n+1.)/2.) * (x+n)**(-(n+1.)/2.-1.)') )
except ImportError:
    gaussian = (lambda x: np.exp(-.5 * x), lambda x: -.5 * np.exp(-.5 * x))
    quartic = (lambda x: np.where(x<1, (1. - x)**2, np.zeros_like(x)), lambda x: np.where(x<1, -2. * (1. - x), np.zeros_like(x)))
    student_n = (lambda x, n: (1. + x/n)**(-(n+1.)/2.), lambda x, n: -(n+1.)/2. * n**((n+1.)/2.) * (x+n)**(-(n+1.)/2.-1.) )

student_1 = (lambda x: student_n[0](x, 1), lambda x: student_n[1](x, 1))
student_2 = (lambda x: student_n[0](x, 2), lambda x: student_n[1](x, 2))
student_3 = (lambda x: student_n[0](x, 3), lambda x: student_n[1](x, 3))
student_9 = (lambda x: student_n[0](x, 9), lambda x: student_n[1](x, 9))


class UKR(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Unsupervised Kernel Regression (UKR)

    Parameters
    ----------
    n_components : int
        Manifold dimension, usually in {1,2,3}.
    kernel : str or tuple(k : func(x), k_der : func(x))
        UKR kernel `k` and its derivative `k_der`. A few examples are included
        in this module: gaussian, quartic and student_{1,2,3,9}.
    metric : {L1, L2} or float
        Distance metric.
        L1: cityblock/manhattan; L2: euclidean
        float : arbitrary Minkowsky
    n_iter : int
        Maximum number of iterations for training the UKR model.
    lko_cv : int
        Leave-k-out cross validation for training the UKR model.
    embeddings : list of initial manifold generators
        If None, the initial embedding is set to PCA and MDS.
        Good choices are:
        * sklean.decomposition.PCA(`n_components`)
        * sklean.decomposition.KernelPCA(`n_components`, kernel='rbf')
        * sklearn.manifold.locally_linear.LocallyLinearEmbedding(n_neighbors, `n_components`, method='modified')
        * sklearn.manifold.MDS(n_components=`n_components`, n_jobs=-1),
        * sklearn.manifold.TSNE(n_components=`n_components`),
    verbose : bool
        Print additional information esp. during the training stage.

    Attributes
    ----------
    X : np.ndarray, shape=(N,D)
        High-dimensional point list for UKR training.
    Y : np.ndarray, shape=(N,n_components)
        Low-dimensional respresentation of `X`.
    """

    def __init__(self, n_components=2, kernel=gaussian, metric='L2', lko_cv=1, n_iter=1000, embeddings=None, verbose=True):
        if isinstance(kernel, basestring):
            if kernel.lower() == 'gaussian':
                self.k, self.k_der = gaussian
            elif kernel.lower() == 'quartic':
                self.k, self.k_der = quartic
            elif kernel.lower() == 'student_1':
                self.k, self.k_der = student_1
            elif kernel.lower() == 'student_2':
                self.k, self.k_der = student_2
            elif kernel.lower() == 'student_3':
                self.k, self.k_der = student_3
            elif kernel.lower() == 'student_9':
                self.k, self.k_der = student_9
        else:
            self.k, self.k_der = kernel

        if isinstance(metric, basestring):
            assert metric in ['L1', 'L2'], "failed condition: metric in ['L1', 'L2']"
            if metric == 'L1': self.metric = 1.
            elif metric == 'L2': self.metric = 2.
        else:
            self.metric = metric

        self.n_components = n_components
        self.lko_cv = lko_cv
        self.n_iter = n_iter
        self.verbose = verbose

        if embeddings is None:
            self.embeddings = [
                    decomposition.PCA(n_components=self.n_components),
                    ## decomposition.KernelPCA(n_components=self.n_components, kernel='poly'),
                    manifold.MDS(n_components=self.n_components, n_jobs=-1),
                    ## manifold.TSNE(n_components=self.n_components),
                    ]

        self.X = None
        self.Y = None
        self.B = None

        pass

    def fit(self, X, y=None):
        """Train the UKR model.

        Parameters
        ----------
        X : np.ndarray, shape=(N,D)
            Sample set with `N` elements and `D` dimensions.

        Returns
        -------
        UKR model object.
        """
        ###########################
        # find an initial embedding

        Y = None
        embed_ = None
        error = np.inf

        for embeddingI, embedding in enumerate(self.embeddings):
            if self.verbose:
                print 'Try embedding %2d/%2d: %s' % (embeddingI+1, len(self.embeddings), embedding.__class__.__name__)

            try:
                Y_init_ = embedding.fit_transform(X)
            except:
                continue

            # normalize initial hypothesis to Y.T * Y = I
            Y_init_ = np.linalg.pinv(np.cov(Y_init_.T)).dot(Y_init_.T).T

            # optimze the scaling factor by using least squares
            def residuals(p, X_, Y_):
                B, P = ukr_bp(Y_ * p, self.k, self.k_der, self.lko_cv, metric=self.metric)
                E = (X_ - ukr_project(X, B)).flatten()
                return E
            p0 = np.ones((1,self.n_components))
            plsq = leastsq(residuals, p0, args=(X, Y_init_))
            Y_init_ = Y_init_ * plsq[0]

            # final projection error estimation
            B, P = ukr_bp(Y_init_, self.k, self.k_der, self.lko_cv, metric=self.metric)
            err_ = ukr_E(X, B)

            if self.verbose:
                print ' Error: %f' % err_

            # store the results if they're an improvement
            if err_ < error:
                error = err_
                Y = Y_init_
                embed_ = embedding

        # Summary:
        if self.verbose:
            print '=> using embedding', embed_.__class__.__name__

        ######################
        # Refine the UKR model

        iRpropPlus = rprop.iRpropPlus()

        for iter in xrange(self.n_iter):
            if self.verbose and iter % 10 == 0:
                print 'UKR iter %5d, Err=%9.6f' % (iter, iRpropPlus.E_prev)

            # derivative of X_model w.r.t. to the error gradient
            B, P = ukr_bp(Y, self.k, self.k_der, self.lko_cv, metric=self.metric,
                    exagg=[1., 4.][iter<self.n_iter/10])
            dY = ukr_dY(Y, X, B, P)

            # reconstruction error
            E_cur = ukr_E(X, B)

            Y = iRpropPlus.update(Y, dY, E_cur)

            ## # re-organize embedded samples with low density: avoid local minima
            ## # this also avoids vast outliers
            ## if iter == self.n_iter * 1/4:
                ## if self.verbose: print 'UKR: re-organize embedded samples'
                ## part = .2
                ## I = np.zeros((Y.shape[0],), dtype=bool)
                ## I[np.random.permutation(range(Y.shape[0]))[:int(part * Y.shape[0])]] = True
                ## Y[I] = ukr_backproject_particles(Y[~I], X[~I], self.k, self.k_der, self.metric, X[I],
                        ## n_particles=100, n_iter=200)

        # store training results
        self.X = X # original data
        self.Y = Y # manifold points

        return self

    def fit_transform(self, X, y=None):
        """Train the UKR model and return the low-dimensional samples.

        Parameters
        ----------
        X : np.ndarray, shape=(N,D)
            Sample set with `N` elements and `D` dimensions.

        Returns
        -------
        Y : np.ndarray, shape=(N, `n_components`)
            Low-dimensional representation of `X`.
        """
        self.fit(X, y)
        return self.Y

    def transform(self, X):
        """Project each sample in `X` to the embedding.
        Uses a particle set for the optimization.

        Parameters
        ----------
        X : np.ndarray, shape=(N,D)
            Sample set with `N` elements and `D` dimensions.

        Returns
        -------
        Y : np.ndarray, shape=(N, `n_components`)
            Low-dimensional representation of `X`.
        """
        Y = ukr_backproject_particles(self.Y, self.X, self.k, self.k_der, self.metric, X,
                n_particles=100, n_iter=200)
        return Y

    def predict(self, Y):
        """Project a set of manifold points into the orignal space.

        Parameters
        ----------
        Y : np.ndarray, shape=(N,`n_components`)
            Arbitrary points on the manifold.

        Returns
        -------
        X : np.ndarray, shape=(N,D)
            Corresponding samples in the high-dimensional space.
        """
        assert self.Y is not None, "untrained UKR model"
        assert Y.shape[1] == self.n_components, \
                "failed condition: Y.shape[1] == self.n_components"

        B, _ = ukr_bp(self.Y, self.k, self.k_der, diagK=-1, Y=Y, metric=self.metric)
        return ukr_project(self.X, B)

    def predict_proba(self, Y):
        """Kernel density estimate for each sample.

        Parameters
        ----------
        Y : np.ndarray, shape=(N,`n_components`)
            Arbitrary points on the manifold.

        Returns
        -------
        p : array-like, shape=(N,)
            Estimated density value for each sample.
        """
        assert self.Y is not None, "untrained UKR model"
        assert Y.shape[1] == self.n_components, \
                "failed condition: Y.shape[1] == self.n_components"

        B, _ = ukr_bp(self.Y, self.k, self.k_der, diagK=-1, Y=Y, bNorm=False, metric=self.metric)
        return B.mean(axis=0)

    pass
