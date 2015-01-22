#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unsupervised Kernel Regression (UKR) for Python.
Implemented as a scikit-learn module.

Author: Christoph Hermes
Created on Januar 16, 2015  18:48:22
"""
import numpy as np
from scipy.optimize import leastsq
import sklearn
from sklearn import datasets, decomposition

# own modules
from ukr_core import ukr_bp, ukr_dY, ukr_E, ukr_project
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
    metric : {L1, L2}
        Distance metric. L1: cityblock/manhattan; L2: euclidean
    n_iter : int
        Maximum number of iterations for training the UKR model.
    lko_cv : int
        Leave-k-out cross validation for training the UKR model.
    embeddings : list of initial manifold generators
        If None, the initial embedding is set to sklean.decomposition.PCA
        Other good choices are:
        * sklean.decomposition.PCA(`n_components`)
        * sklearn.manifold.locally_linear.LocallyLinearEmbedding(n_neighbors, `n_components`, method='modified')
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

        self.n_components = n_components
        self.metric = metric
        self.lko_cv = lko_cv
        self.n_iter = n_iter
        self.verbose = verbose

        if embeddings is None:
            self.embeddings = [decomposition.PCA(n_components=self.n_components)]

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
        ## B_ = None
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
            Y_init_ = (Y_init_ - Y_init_.mean(axis=0)) / Y_init_.std(axis=0)

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
                ## B_ = B
                embed_ = embedding

        # Summary:
        if self.verbose:
            print 'Using embedding', embed_.__class__.__name__
            print ' Error: %f' % error

        ######################
        # Refine the UKR model

        iRpropPlus = rprop.iRpropPlus()

        for iter in xrange(self.n_iter):
            if self.verbose and iter % 10 == 0:
                print 'UKR iter %5d, Err=%9.6f' % (iter, iRpropPlus.E_prev)

            # derivative of X_model w.r.t. to the error gradient
            B, P = ukr_bp(Y, self.k, self.k_der, self.lko_cv, metric=self.metric)
            dY = ukr_dY(Y, X, B, P)

            # reconstruction error
            E_cur = ukr_E(X, B)

            Y = iRpropPlus.update(Y, dY, E_cur)

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
        raise NotImplementedError('To come.')

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


if __name__ == '__main__':
    from datetime import datetime
    import matplotlib.pyplot as plt
    import itertools

    ds_name = 'iris' # {iris, digits}

    if ds_name == 'iris':
        ds = datasets.load_iris()
        X = ds.data
        # make each column equal to the distance metric
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        lko_cv = 3
        max_iter = 3000
    elif ds_name == 'digits':
        ds = datasets.load_digits(n_class=10)
        X = ds.data

        lko_cv = 10
        max_iter = 20000
    y = ds.target
    q = 2
    kernel = student_3

    u = UKR(n_components=q, kernel=kernel, n_iter=max_iter, lko_cv=lko_cv, metric='L2')
    mani = u.fit_transform(X)

    f = plt.figure(1, figsize=(8*3,5*3))

    ax = f.add_subplot(121)
    clrs = itertools.cycle('rgbcykm')
    mrks = itertools.cycle('.x+')
    for y_ in np.unique(y):
        ax.plot(mani[y==y_,0], mani[y==y_,1], clrs.next() + mrks.next(), label=str(ds.target_names[y_]))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.legend(loc='best')

    # visualize density
    XX, YY = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 200),
            np.linspace(ylim[0], ylim[1], 200))
    dens = u.predict_proba(np.c_[XX.flatten(), YY.flatten()]).reshape(XX.shape)

    ax = f.add_subplot(122)
    ax.plot(mani[:,0], mani[:,1], 'g.')
    ax.contour(XX, YY, np.log(dens + 1), 15)

    ## plt.show()
    tm = datetime.now().strftime('%y%m%d_%H%M%S_%f')[:-3]
    plt.savefig('ukr_%s_%s.png' % (ds_name, tm), bbox_inches='tight')
