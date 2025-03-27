#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def variance(data):
    return data.var()

def median(data):
    return data.median()

def mean(data):
    return data.mean(axis=0)

def std(data):
    return data.std()

def rms(data):
    return np.sqrt(np.mean(np.square(data)))

def zero_crossing(data):
    zc = (np.diff(np.signbit(data), axis=0) != 0).sum()
    return zc

def sum_of_squares(data):
    return np.sum(np.square(data))

def covariance(data):
    return data.cov()


