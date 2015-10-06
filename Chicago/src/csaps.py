# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 20:05:13 2015

@author: Kevin
"""

import numpy as np
import scipy as sp
from scipy import interpolate
from scipy.sparse import linalg


def csaps(x,y,p,xi=[],w=[]):
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
    
    coefs =   np.concatenate((dd.reshape((1,dd.size)) / 6, 
                              cc.reshape((1,cc.size)) / 2, 
                              bb.reshape((1,bb.size)), 
                              aa.reshape((1,aa.size))))                      
    ret = interpolate.interpolate.PPoly(coefs, x)

    xi = np.array(xi)
    if xi.size != 0:
        ret = ret.__call__(xi)
        
    return ret






#  if(columns(x) > 1)
#    x = x.';
#    y = y.';
#    w = w.';
#  endif
#
#  [x,i] = sort(x);
#  y = y(i, :);
#
#  n = numel(x);
#  
#  if isempty(w)
#    w = ones(n, 1);
#  end
#
#  h = diff(x);
#   R = spdiags([h(1:end-1) 2*(h(1:end-1) + h(2:end)) h(2:end)], [-1 0 1], n-2, n-2);
#   QT = spdiags([1 ./ h(1:end-1) -(1 ./ h(1:end-1) + 1 ./ h(2:end)) 1 ./ h(2:end)], [0 1 2], n-2, n);
#  solve for the scaled second derivatives u and for the function values a at the knots (if p = 1, a = y) 
#   u = (6*(1-p)*QT*diag(1 ./ w)*QT' + p*R) \ (QT*y);
#
#  a = y - 6*(1-p)*diag(1 ./ w)*QT'*u;
#
## derivatives at all but the last knot for the piecewise cubic spline
#  aa = a(1:(end-1), :);
#  cc = zeros(size(y)); 
#  cc(2:(n-1), :) = 6*p*u; #cc([1 n], :) = 0 [natural spline]
#  dd = diff(cc) ./ h;
#  cc = cc(1:(end-1), :);
#  bb = diff(a) ./ h - (cc/2).*h - (dd/6).*(h.^2);
#
#  ret = mkpp (x, cat (2, dd'(:)/6, cc'(:)/2, bb'(:), aa'(:)), size(y, 2));
#
#  if ~isempty(xi)
#    ret = ppval (ret, xi);
#  endif
#
#endfunction  
  
  