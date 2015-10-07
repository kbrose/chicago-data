# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 20:05:13 2015

Translation of octave code for CSAPS.

@author: Kevin
"""

import numpy as np
import scipy as sp
from scipy import interpolate
from scipy.sparse import linalg


def csaps(x, y, p, xi=[], w=[]):
    # sort the inputs by ordering of x
    ii = np.argsort(x)
    x = np.array(x)
    y = np.array(y)
    x = x.take(ii)
    y = y.take(ii)

    h = np.diff(x)

    n = np.size(x)

    if np.size(w) == 0:
        w = np.ones([n, 1])

    R = sp.sparse.spdiags(np.array([h[0:-1],
                                    2.*(h[0:-1] + h[1:]),
                                    h[1:]]), [-1, 0, 1], n-2, n-2)

    QT = sp.sparse.spdiags(np.array([1. / h[0:-1],
                                     -(1. / h[0:-1] + 1. / h[1:]),
                                     1. / h[1:]]), [0, -1, -2], n, n-2).transpose()

    # solve for the scaled second derivatives u and
    # for the function values a at the knots (if p = 1, a = y)

    v = 6*(1-p)*QT.dot(sp.sparse.spdiags(1. / w.flatten(), 0, len(w), len(w))).dot(QT.T) + p*R
    u = linalg.spsolve(v, QT.dot(y))
    a = y - 6*(1-p)*sp.sparse.spdiags(1. / w.flatten(), 0, len(w), len(w)).dot(QT.T).dot(u)

    # derivatives at all but the last knot for the piecewise cubic spline
    aa = a[0:-1]
    cc = np.zeros(y.shape)
    cc[1:n-1] = 6 * p * u
    dd = np.diff(cc) / h
    cc = cc[0:-1]
    bb = np.diff(a) / h - cc / 2 * h - dd / 6 * h ** 2

    # shape coefficients and create piece-wise polynomial
    coefs = np.concatenate((dd.reshape((1, dd.size)) / 6,
                            cc.reshape((1, cc.size)) / 2,
                            bb.reshape((1, bb.size)),
                            aa.reshape((1, aa.size))))
    ret = interpolate.interpolate.PPoly(coefs, x)

    # check if we should evaluate the smoothing spline
    xi = np.array(xi)
    if xi.size != 0:
        ret = ret(xi)

    return ret
