#!/usr/bin/env python
from math import log, pi, exp

DATA =   [(0, 1), (0, 0), (1, 0), (2, 2)]
LABELS = [ True,  False,   True,  False ]

def weighted_mean(data, weights):
    return sum(x*w for x, w in zip(data, weights)) / sum(weights)

def weighted_variance(data, weights):
    mu = weighted_mean(data, weights)
    weighted_errors = [w*(x - mu)**2 for x, w in zip(data, weights)]
    return sum(weighted_errors) / sum(weights)

class NaiveBayesClassifier(object):
    """
    Implements Naive Bayes with weighted data
    """
    def __init__(self, data, weights, labels):
        self.models = list()
        for feature in range(len(data[0])):
            values = [data[row][feature] for row in range(len(data))]
            self.models.append(FeatureModel(values, weights, labels))
        self.class_model = ClassModel(weights, labels)

    def classify(self, instance):
        return (self.confidence(instance) >= 0.5)

    def confidence(self, instance):
        probability = dict()
        for label in (True, False):
            # Convert True/False to +1/-1
            log_sum = 0
            for value, model in zip(instance, self.models):
                log_sum += model.log_likelihood(value, label)
            log_sum += self.class_model.log_likelihood(label)
            probability[label] = exp(log_sum)

        return probability[True] / (probability[True] + probability[False])

class FeatureModel(object):
    """
    The model we build of the data
    given observed values
    """
    def __init__(self, values, weights, labels):
        data = zip(values, weights, labels)
        pos_data = [(v, w) for v, w, label in data if label is True]
        neg_data = [(v, w) for v, w, label in data if label is False]

        self.mus = dict()
        self.mus[True] = weighted_mean(*zip(*pos_data))
        self.mus[False] = weighted_mean(*zip(*neg_data))

        self.variances = dict()
        self.variances[True] = weighted_variance(*zip(*pos_data))
        self.variances[False] = weighted_variance(*zip(*neg_data))

    def log_likelihood(self, value, label):
        mu = self.mus[label]
        sigma2 = self.variances[label]
        return -((value - mu)**2)/(2*sigma2) - 0.5*log(2*pi*sigma2)

class ClassModel(object):
    """
    Models the likelihood of classes given
    observed class labels
    """
    def __init__(self, weights, labels):
        self.probs = dict()
        pos_weights = sum(w for w, label in zip(weights, labels)
                            if label is True)
        self.probs[True] = pos_weights / sum(weights)
        self.probs[False] = 1.0 - self.probs[True]

    def log_likelihood(self, label):
        return log(self.probs[label])

def adaboost(data, labels, step=True):
    pass

def main(step):
    nb = NaiveBayesClassifier(DATA, [0.25, 0.25, 0.25, 0.25], LABELS)
    print nb.confidence((1,1))

if __name__ == '__main__':
    from sys import argv
    main('nostep' not in argv)
