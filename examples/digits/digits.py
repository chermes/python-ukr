#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
UKR test with a subset of the MNIST digits data.

Author: Christoph Hermes
Created on Februar 07, 2015  18:50:36


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

# make the UKR module visible to Python
import os, sys
lib_path = os.path.abspath(os.path.join('..', '..', 'src_naive'))
sys.path.append(lib_path)

import time
import itertools

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

import ukr


# load the data into memory
ds = datasets.load_digits(n_class=3)
X = ds.data

lko_cv = 1
max_iter = (ds.target.max() + 1) * 1000
metric = 'L2'
y = ds.target
q = 2
## kernel, kernel_s = ukr.gaussian, 'gaussian'
kernel, kernel_s = ukr.student_3, 'student_3'

tm = time.time()
u = ukr.UKR(n_components=q, kernel=kernel, n_iter=max_iter, lko_cv=lko_cv, metric=metric)
mani = u.fit_transform(X)
print 'UKR training took %.2f seconds' % (time.time() - tm)

f = plt.figure(1, figsize=(8*3,5*3))

if q == 2:
    ax = f.add_subplot(111)
    clrs = itertools.cycle('rgbcykm')
    mrks = itertools.cycle('.x+')
    for y_ in np.unique(y):
        ax.plot(mani[y==y_,0], mani[y==y_,1], clrs.next() + mrks.next(), label=str(ds.target_names[y_]))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.legend(loc='best')

    # visualize density and possibly the digits images, too
    nX, nY = 200, 200
    XX, YY = np.meshgrid(
            np.linspace(xlim[0], xlim[1], nX),
            np.linspace(ylim[0], ylim[1], nY))
    dens = u.predict_proba(np.c_[XX.flatten(), YY.flatten()])
    ax.contour(XX, YY, np.log2(dens.reshape(XX.shape) + 1), 15, alpha=.3)
    ax.axis('equal')

    # visualize the image patch space
    patches_ = u.predict(np.c_[XX.flatten(), YY.flatten()])
    patches = [p.reshape(ds.images[0].shape) * np.log(d+1) for p,d in zip(patches_, dens)]

    oS = ds.images[0].shape
    img = np.zeros((oS[0] * nY + 1, oS[1] * nX + 1))
    for i in range(nY):
        iy = nY * oS[0] - (oS[0] * i) - oS[0] # reversed
        for j in range(nX):
            ix = oS[1] * j
            img[iy:iy + oS[0], ix:ix + oS[1]] = patches[i * nY + j].reshape(oS)

    f = plt.figure(2, figsize=(8*3,5*3))
    plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray_r)

elif q == 3:
    ax = Axes3D(f)
    clrs = itertools.cycle('rgbcykm')
    mrks = itertools.cycle('.x+')
    for y_ in np.unique(y):
        ax.plot3D(mani[y==y_,0], mani[y==y_,1], mani[y==y_,2], clrs.next() + mrks.next(), label=str(ds.target_names[y_]))
    plt.legend(loc='best')

plt.show()
