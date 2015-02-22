#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates, interpolates and visualizes the "monkey head" manifold. The head
is the Blender mascot Suzanne rotating around each three-dimensional axis.

Author: Christoph Hermes
Created on Februar 21, 2015  19:38:28


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

import os, sys
lib_path = os.path.abspath(os.path.join('..', '..', 'src_naive'))
sys.path.append(lib_path)

import glob
import numpy as np
import scipy.ndimage as ndi
import sklearn.manifold
import sklearn.decomposition

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import ukr

# load the monkey head images into memory
D = sorted(glob.glob('images_rotatingMonkey/*.png'))
X_raw = np.zeros((len(D), np.prod(ndi.imread(D[0]).shape)))
for imgI, img in enumerate(D):
    X_raw[imgI] = ndi.imread(img).astype(np.float64).flatten()
N = X_raw.shape[0]

initEmbed = sklearn.decomposition.PCA(3)
model = ukr.UKR(n_components=3, kernel=ukr.student_k(3), n_iter=1000, embeddings=[initEmbed], metric=2, enforceCycle=True)
mani = model.fit_transform(X_raw)

f = plt.figure()
f.clf()
ax = Axes3D(f)
ax.plot3D(mani[:N/3,0], mani[:N/3,1], mani[:N/3,2], '.-', label='pitch')
ax.plot3D(mani[N/3:N*2/3,0], mani[N/3:N*2/3,1], mani[N/3:N*2/3,2], '.-', label='yaw')
ax.plot3D(mani[N*2/3:,0], mani[N*2/3:,1], mani[N*2/3:,2], '.-', label='roll')
plt.legend()
plt.show()
