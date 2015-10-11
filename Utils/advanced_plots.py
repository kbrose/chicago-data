# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:37:10 2015

@author: Kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

# taken from http://stackoverflow.com/questions/7941207/
# modified to plot colums instead of rows
# modified to plot histograms down the diagonal if specified
def scatterplot_matrix(data, names=[], hist=True, **kwargs):
    """
    Plots a scatterplot matrix of subplots.  Each column of "data" is plotted
    against other columns, resulting in a ncols by ncols grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containing the subplot grid.
    """
    numdata, numvars = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    if not kwargs.has_key('linestyle'):
        kwargs['linestyle'] = 'none'
    if not kwargs.has_key('marker'):
        kwargs['marker'] = '.'

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            # FIX #1: this needed to be changed from ...(data[x], data[y],...)
            axes[x,y].plot(data[:,y], data[:,x], **kwargs)

    # Label the diagonal subplots...
    if not names:
        names = ['var '+str(i) for i in range(numvars)]

    if hist:
        for i, label in enumerate(names):
            histogram(data[:,i], ax=axes[i,i])
            #print(dir(axes[0,i]))
#            is_bot = ((i + 1) % 2)
#            axes[0 - is_bot,i].set_xlabel(label)
#            if not is_bot:
#                axes[0,i].xaxis.set_label_position('top')
    else:
        for i, label in enumerate(names):
            axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[j,i].set_xlabel(names[i])

        axes[i,j].yaxis.set_visible(True)
        axes[i,j].set_ylabel(names[i])

        if j == 0:
            axes[j,i].xaxis.set_label_position('top')
        else:
            axes[i,j].yaxis.set_label_position('right')

    # FIX #2: if numvars is odd, the bottom right corner plot doesn't have the
    # correct axes limits, so we pull them from other axes
    if numvars % 2:
        yticks = axes[-1,0].get_yticks()

        currticks = axes[-1,-1].get_yticks()
        newticks = np.linspace(min(currticks), max(currticks), len(yticks))
        axes[-1,-1].set_yticks(newticks)
        axes[-1,-1].set_yticklabels(__massage_ticklabels(yticks))

    return fig


def histogram(data, ax=None, **kwargs):
    """
    A convenience wrapper for plt.hist(). This function automatically specifies
    an optimal number of bins according to the Freedman-Diaconis rule of thumb:
        bins = (max-min)/(2. ∗ IQR ∗ n**(−1./3))
    where
        max is the maximum value of the data set
        min is the minimum value of the data set
        IQR is the interquartile range (75th percentile - 25th percentile)
        n is the number of data points

    In addition, if the data is multi-dimensional, then proper specifications
    are passed, including alpha=.5 and stepfilled=True. Additionally, the edges
    are specified to be the same for each column.

    An optional argument ax can be specified which controls where the
    histogram is plotted. This defaults to gca() if not specified.
    """
    if ax is None:
        ax = plt.gca()
    if type(data) is not np.ndarray:
        data = np.array(data)
    shape = data.shape
    if len(shape) > 1 and min(shape) > 1:
        alpha = .8
        histtype = 'stepfilled'
        N = shape[0]
    else:
        alpha = 1
        histtype = 'bar'
        N = len(data)

    x = data.flatten()
    nbins = (max(x) - min(x)) / np.diff(np.percentile(x, [25, 75]))
    nbins = nbins / (2. * N ** (-1./3))
    edges = np.linspace(min(x), max(x), nbins+1)

    ax.hist(data, bins=edges, histtype=histtype, alpha=alpha, **kwargs)


def __massage_ticklabels(labels):
    """
    Take an array of numbers and make turn them into strings that
    are more presentable for tick labels.
    """
    ret_labels = []
    r = max(labels) - min(labels)
    tol = r * .001 / len(labels)
    for label in labels:
        if label - np.floor(label) < tol or np.ceil(label) - label < tol:
            ret_labels.append(str(int(np.round(label))).replace(u'-', u'\u2212'))
        else:
            ret_labels.append(str(np.round(label,4)).replace(u'-', u'\u2212'))
    return ret_labels
