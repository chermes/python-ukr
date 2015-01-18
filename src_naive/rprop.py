#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
RProp for Python/Numpy and its variants.

Author: Christoph Hermes
Created on November 23, 2014  12:30:54
"""
import numpy as np

import logging as log
log.basicConfig()
logging = log.getLogger('rprop')
logging.setLevel(log.DEBUG)


class iRpropPlus:
    """iRprop+ optimization for Python/Numpy.

    Parameters
    ----------
    eta_p : float
        Acceleration factor in the gradient descent direction.
    eta_m : float
        Deceleration factor in the gradient descent direction.
    eta_min : float
    eta_max : float
    dR_init : float
    verbose : bool

    References
    ----------
    C. Igel and M. Huesken: Improving the Rprop Learning Algorithm
    Proceedings of the Second International Symposium on Neural Computation, 2000
    """

    def __init__(self, eta_p=1.2, eta_m=.7, eta_max=50, eta_min=1e-6, dR_init=0.02, verbose=0):
        self.eta_p = eta_p
        self.eta_m = eta_m
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.dR_init = dR_init

        self.iter = 0
        self.verbose = verbose

        self.dR_1 = None
        self.E_prev = np.inf
        self.eta_1 = None
        self.dW_prev = None
        pass

    def update(self, X, dX, E_cur):

        if self.iter == 0:
            self.dR_1 = np.zeros_like(X)
            self.eta = np.ones_like(X) * self.dR_init
            self.eta_1 = np.ones_like(X) * self.dR_init
            self.dW_prev = np.zeros_like(X)

        I = (self.dR_1 * dX) > 1e-10 # accelerate
        if I.sum() > 0:
            if self.verbose > 0:
                logging.debug('accelerate: %d', I.sum())
            self.eta[I] = self.eta_1[I] * self.eta_p
            self.eta[self.eta > self.eta_max] = self.eta_max
            #
            dW = -np.sign(dX[I]) * self.eta[I]
            X[I] = X[I] + dW
            #
            self.dW_prev[I] = dW
            self.eta_1[I] = self.eta[I]
        I = (self.dR_1 * dX) < -1e-10 # decelerate
        if I.sum() > 0:
            if self.verbose > 0:
                logging.debug('decelerate: %d', I.sum())
            self.eta[I] = self.eta_1[I] * self.eta_m
            self.eta[self.eta < self.eta_min] = self.eta_min
            #
            if E_cur > self.E_prev:
                X[I] = X[I] - self.dW_prev[I]
            dX[I] = 0.
            #
            self.eta_1[I] = self.eta[I]
        I = np.abs(self.dR_1 * dX) <= 1e-10 # switch directions
        if I.sum() > 0:
            if self.verbose > 0:
                logging.debug('switch directions: %d', I.sum())
            dW = -np.sign(dX[I]) * self.eta[I]
            X[I] = X[I] + dW
            #
            self.dW_prev[I] = dW

        self.dR_1 = dX.copy()
        self.E_prev = E_cur
        self.iter += 1

        return X

    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test the Rprop algorithms via the Rosenbrock function
    # https://en.wikipedia.org/wiki/Rosenbrock_function
    # The global optimum is at (a,a^2) with F(x,y)=0
    a, b = 1., 100.
    F = lambda x, y: (a-x)**2 + b*(y-x**2)**2
    dF_x = lambda x, y: 4.*b*x**3 - 2.*x*(2.*b*y-1) - 2.*a
    dF_y = lambda x, y: 2.*b*y - 2.*b*x**2

    # starting point
    X_init = np.array([-.5, 2.], dtype=np.float64) # combined (x,y)

    max_iter = 2000
    X_coll = np.zeros((max_iter, 2))

    rprop = iRpropPlus(verbose=1)
    X = X_init.copy()
    for i in range(max_iter):
        dX = np.array([dF_x(X[0], X[1]), dF_y(X[0], X[1])])
        E_cur = F(X[0], X[1])
        X = rprop.update(X, dX, E_cur)
        X_coll[i,:] = X
        print i, F(X[0], X[1]), X

    sur = 2
    XX, YY = np.meshgrid(
            np.linspace(min(X_init[0],a)-sur,max(X_init[0],a)+sur, 200),
            np.linspace(min(X_init[1],a**2)-sur, max(X_init[1],a**2)+sur, 200))
    plt.figure(1)
    plt.contour(XX, YY, np.log(F(XX, YY)+1), 20)
    plt.plot(a, a**2, 'ro')
    plt.text(a, a**2, 'target')
    plt.plot(X_coll[:,0], X_coll[:,1], 'b-')
    plt.plot(X_coll[:,0], X_coll[:,1], 'b.')
    plt.text(X_init[0], X_init[1], 'start')
    plt.show()

    import IPython; IPython.embed()
