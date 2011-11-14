#!/usr/bin/env python
from math import sqrt

def weighted_mean(data, weights):
    return sum(x*w for x, w in zip(data, weights)) / sum(weights)

def weighted_std(data, weights):
    mu = weighted_mean(data, weights)
    weighted_errors = [w*(x - mu)**2 for x, w in zip(data, weights)]
    return sqrt(sum(weighted_errors) / sum(weights))
