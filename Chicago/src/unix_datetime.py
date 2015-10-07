# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 23:37:50 2015

from http://stackoverflow.com/questions/13260863/

"""

import calendar, datetime

# Convert a unix time u to a datetime object d, and vice versa
def dt(u):
    if type(u) == list:
        return [datetime.datetime.utcfromtimestamp(x) for x in u]
    return datetime.datetime.utcfromtimestamp(u)
def ut(d):
    if type(d) == list:
        return [calendar.timegm(x.timetuple()) for x in d]
    return calendar.timegm(d.timetuple())
    