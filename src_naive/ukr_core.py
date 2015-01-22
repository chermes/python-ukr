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


def zero_out_diag(M, lko):
    """Zero out a diagonal band of the matrix `M` defined by `lko`.

    Notes
    -----
    The input matrix M is changed!
    """
    if lko >= 0:
        for k_ in np.arange(-lko, lko+1):
            M = M - np.diag(np.diagonal(M, k_), k_)

    return M


def ukr_bp(Y_model, k, k_der, diagK=-1, Y=None, bNorm=True, metric='L2'):
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
        Normalize the kernel matrices?
    metric : {L1, L2}
        Distance metric.

    Returns
    -------
    B : np.ndarray, shape=(N,N)
        Kernelized distance matrix between the samples `Y_model` and `Y`.
        Normalized such that the columns sum to 1.
    P : np.ndarray, shape=(N,N)
        Derivative of the kernelized distance matrix between the samples
        `Y_model` and `Y`.
    """
    assert metric in ['L1', 'L2'], "failed condition: metric in ['L1', 'L2']"

    if Y is None:
        Y = Y_model

    if metric == 'L2':
        # pairwise distances, sqared L2 norm: (Y_model - Y)^2
        Y1 = (Y_model**2).sum(axis=1)
        Y2 = (Y**2).sum(axis=1)
        D = ((-2. * Y_model.dot(Y.T) + Y2).T + Y1).T
    elif metric == 'L1':
        # squared L1 norm
        ## D = (Y_model[np.newaxis, :, :] - Y[:, np.newaxis, :]).sum(2)**2
        D = distance.cdist(Y_model, Y, 'cityblock')**2

    # LOO CV: zero out the diagonal elements
    K = zero_out_diag(k(D), diagK)

    Bsum = K.sum(axis=0)
    if bNorm:
        Bsum[np.abs(Bsum)<1e-16] = 1e-16 # prevent zeros
        B = K / Bsum # normalize by the row-wise kernel sums
    else:
        B = K

    K_der = zero_out_diag(k_der(D), diagK)
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
