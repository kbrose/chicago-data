# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 23:37:50 2015

from http://stackoverflow.com/questions/13260863/

"""

import calendar, datetime

# Convert a unix time u to a datetime object d, and vice versa
def dt(u): return datetime.datetime.utcfromtimestamp(u)
def ut(d): return calendar.timegm(d.timetuple())
    