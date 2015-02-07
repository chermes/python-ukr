#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
UKR core methods, Python/Numpy version.
These methods are of naive kind, i.e. directly programmed after the
paper/thesis instructions and non-optimized.

Author: Christoph Hermes
Created on November 20, 2014  19:14:30
"""
import numpy as np
from scipy.spatial import distance


def zero_out_diag(M, lko, zband=None):
    """Zero out a diagonal band of the matrix `M` defined by `lko`.

    Notes
    -----
    The input matrix M is changed!
    """
    if lko >= 0:
        if zband is None:
            zband = np.ones(M.shape, dtype=bool)
            zband[np.triu_indices_from(zband, lko)] = False
        M[~(zband[::-1,::-1] - zband)] = 0

    return M, zband


def ukr_bp(Y_model, k, k_der, diagK=-1, Y=None, bNorm=True, metric=2., exagg=1.):
    """Kernelized pairwise distances and its derivatives.

    Parameters
    ----------
    Y_model : np.ndarray, shape=(N,q)
        Support manifold points corresponding to N `Y` elements.
    k, k_der : func(x)
        Kernel and its derivative.
    diagK : int
        Leave-`diagK`-out cross validation parameter.
        -1: do not use the parameter.
    Y : np.ndarray, shape=(M,q), optional
        Project these manifold points to the original dimension. If None, the
        Y_model set is used.
    bNorm : bool
        Normalize the kernel matrices to column-wise sum=1?
    metric : float
        Distance coefficient of the Minkowsky metric.
    exagg : float
        Exaggeration factor for the distance matrix.

    Returns
    -------
    B : np.ndarray, shape=(N,N)
        Kernelized distance matrix between the samples `Y_model` and `Y`.
        Normalized such that the columns sum to 1.
    P : np.ndarray, shape=(N,N)
        Derivative of the kernelized distance matrix between the samples
        `Y_model` and `Y`.
    """

    if Y is None:
        Y = Y_model

    if np.abs(metric - 2) < 1e-5:
        # pairwise distances, sqared L2 norm: (Y_model - Y)^2
        Y1 = (Y_model**2).sum(axis=1)
        Y2 = (Y**2).sum(axis=1)
        D = ((-2. * Y_model.dot(Y.T) + Y2).T + Y1).T
    elif np.abs(metric - 1) < 1e-5:
        # squared L1 norm
        ## D = (Y_model[np.newaxis, :, :] - Y[:, np.newaxis, :]).sum(2)**2
        D = distance.cdist(Y_model, Y, 'cityblock')**2
    else:
        D = distance.cdist(Y_model, Y, 'minkowski', metric)**2
    
    D = D * exagg

    K = k(D)
    # LOO CV: zero out the diagonal elements
    K, zband = zero_out_diag(K, diagK)

    Bsum = K.sum(axis=0)
    if bNorm:
        Bsum[np.abs(Bsum)<1e-16] = 1e-16 # prevent zeros
        B = K / Bsum # normalize by the row-wise kernel sums
    else:
        B = K

    K_der = k_der(D)
    K_der, _ = zero_out_diag(K_der, diagK, zband)
    if bNorm:
        P = -2. * K_der / Bsum
    else:
        P = -2. * K_der
    ## P[np.abs(P)<1e-16] = 0.

    return B, P


def ukr_dY(Y_model, X, B, P):
    """Derivatives of Y_model w.r.t. the reconstruction error gradient."""

    ## L = np.ones((Y_model.shape[0],1))
    ## n = Y_model.shape[0]

    M = X.dot(X.T.dot(B) - X.T)
    ## M_ = X.dot(X.T.dot(B - np.eye(n))) # equal but slower due to eye constr.
    ## Q = P * M - P * (M*B).sum(axis=0)
    Q = P * (M - (M*B).sum(axis=0)) # this is slightly faster
    ## Q[np.abs(Q) < 1e-16] = 0
    ## dY = (Y_model.T.dot(Q + Q.T - np.diag(L.T.dot(Q+Q.T)[0])) * 2./B.shape[0]).T
    dY = (Y_model.T.dot(Q + Q.T - np.diag((Q+Q.T).sum(axis=0))) * 2./B.shape[0]).T
    ## dY[np.abs(dY) < 1e-16] = 0

    return dY


def ukr_E(X, B):
    """UKR reconstruction error."""
    E = ((X - B.T.dot(X))**2).sum() / B.shape[0] # (Frobenius norm)^2

    return E


def ukr_project(X, B):
    return B.T.dot(X)


def sus(fitness, nSel):
    """Stochastic Universal sampling
    # TODO: ref
    """

    I = np.argsort(fitness)[::-1]
    C = np.cumsum(fitness[I])

    F = float(fitness.sum())
    P = F / nSel
    r = np.random.rand() * P

    idxs = np.array([I[np.nonzero(C >= r + P*i)[0][0]] for i in range(nSel)])

    return idxs


def ukr_backproject_particles(Y_model, X_model, k, k_der, metric, X, n_particles=100, n_iter=100):
    """Project high-dimensional points `X` to the embedding using particles.
    """
    X = np.atleast_2d(X)
    Y = np.zeros((X.shape[0], Y_model.shape[1]))

    # for each sample X
    for iX in range(X.shape[0]):
        print 'UKR_core: optimize sample %d of %d' % (iX + 1, X.shape[0])
        # init particle set PY
        D = distance.cdist(X_model, [X[iX]], 'minkowski', metric).flatten()
        PY = Y_model[np.argsort(D)[:n_particles]]

        for j in range(n_iter):
            # determine fitness `F` for each particle: reconstruction error
            B, _ = ukr_bp(Y_model, k, k_der, Y=PY, metric=metric, bNorm=False)
            occProb = (B.mean(axis=0) + 1e-10) # occurrence probability
            B = B / (B.sum(axis=0) + 1e-10) # post-normalization
            F = 1. / (np.sqrt(((ukr_project(X_model, B) - X[iX])**2).sum(axis=1)) + 1e-10)
            F = F * occProb # down-weight particles outside the manifold

            # select new particle set
            PY = PY[sus(F, n_particles)]

            # randomize particle set with linear annealing
            PY = PY + np.random.randn(PY.shape[0], PY.shape[1]) * D.std() * float((n_iter - j) / n_iter)

        # estimate final position
        Y[iX] = np.median(PY, axis=0)

    return Y
