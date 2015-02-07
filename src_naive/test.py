#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
UKR test.

Author: Christoph Hermes
Created on Februar 07, 2015  18:50:36
"""

import time
import itertools

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

import ukr


ds_name = ['iris', 'digits'][1]

if ds_name == 'iris':
    ds = datasets.load_iris()
    X = ds.data
    # make each column equal w.r.t. the distance metric
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    lko_cv = 1
    max_iter = 500
    metric = 2
elif ds_name == 'digits':
    ds = datasets.load_digits(n_class=4)
    X = ds.data

    lko_cv = 1
    max_iter = 2000
    metric = 'L2'
y = ds.target
q = 2
kernel, kernel_s = ukr.gaussian, 'gaussian'
## kernel, kernel_s = ukr.student_3, 'student_3'

tm = time.time()
u = ukr.UKR(n_components=q, kernel=kernel, n_iter=max_iter, lko_cv=lko_cv, metric=metric)
mani = u.fit_transform(X)
print 'UKR training took %.2f seconds' % (time.time() - tm)

f = plt.figure(1, figsize=(8*3,5*3))

if q == 2:
    ax = f.add_subplot(121)
    clrs = itertools.cycle('rgbcykm')
    mrks = itertools.cycle('.x+')
    for y_ in np.unique(y):
        ax.plot(mani[y==y_,0], mani[y==y_,1], clrs.next() + mrks.next(), label=str(ds.target_names[y_]))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.legend(loc='best')

    # visualize density
    try: # cope with non-UKR manifolds
        XX, YY = np.meshgrid(
                np.linspace(xlim[0], xlim[1], 200),
                np.linspace(ylim[0], ylim[1], 200))
        dens = u.predict_proba(np.c_[XX.flatten(), YY.flatten()]).reshape(XX.shape)

        ax = f.add_subplot(122)
        ax.plot(mani[:,0], mani[:,1], 'g.')
        ax.contour(XX, YY, np.log(dens + 1), 15)
    except AttributeError:
        pass

elif q == 3:
    ax = Axes3D(f)
    clrs = itertools.cycle('rgbcykm')
    mrks = itertools.cycle('.x+')
    for y_ in np.unique(y):
        ax.plot3D(mani[y==y_,0], mani[y==y_,1], mani[y==y_,2], clrs.next() + mrks.next(), label=str(ds.target_names[y_]))
    plt.legend(loc='best')

plt.show()
## from datetime import datetime
## tm = datetime.now().strftime('%y%m%d_%H%M%S_%f')[:-3]
## plt.savefig('ukr_%s_%s_%s_%s_lko%02d.png' % (ds_name, tm, kernel_s, metric, lko_cv), bbox_inches='tight')
