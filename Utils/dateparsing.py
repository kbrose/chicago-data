# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 10:56:27 2015

@author: Kevin
"""

import pandas as pd

def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.

    Thanks to fixxxer, found at
    http://stackoverflow.com/questions/29882573
    """
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.apply(lambda v: dates[v])
