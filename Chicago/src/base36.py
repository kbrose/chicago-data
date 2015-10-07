# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 23:02:05 2015

Taken from http://stackoverflow.com/questions/1181919/python-base-36-encoding

@author: Kevin
"""


def base36encode(numberInDecimal):
    if not isinstance(numberInDecimal, (int, long)):
        raise TypeError('number must be an integer')
    if numberInDecimal < 0:
        raise ValueError('number must be positive')

    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    base36 = ''
    while numberInDecimal:
        numberInDecimal, i = divmod(numberInDecimal, 36)
        base36 = alphabet[i] + base36

    return base36 or alphabet[0]


def base36decode(numberInBase36):
    return int(numberInBase36, 36)
