# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:59:48 2020

@author: hugov
"""


def x_in_y(x, y):
    # check if x is a nested list
    if any(isinstance(i, list) for i in x):
        return all((any((set(x_).issubset(y_) for y_ in y)) for x_ in x))
    else:
        return any((set(x).issubset(y_) for y_ in y))