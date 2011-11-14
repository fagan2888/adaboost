#!/usr/bin/env python
from math import log, pi, exp
from itertools import count

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

class AdaboostClassifier(object):
    """
    Combines individual weighted classifiers
    """
    def __init__(self, classifiers, alphas):
        self.classifiers = classifiers
        Z = sum(alphas)
        self.alphas = [a / Z for a in alphas]

    def classify(self, instance, verbose=True):
        lines = list()
        avg = 0
        first = True
        for h, a in zip(self.classifiers, self.alphas):
            if first:
                indent = 'f(x) = '
                first = False
            else:
                indent = '       '
            cls = 2*h.classify(instance) - 1
            avg += a*cls
            lines.append(indent + ('%.5f * (%2d)' % (a, cls)))
        print ' +\n'.join(lines)
        print avg
        return (avg >= 0)

def adaboost(data, labels, verbose=True, step=True):
    #Initialize weights
    weights = [1.0 / len(data) for i in range(len(data))]
    classifiers = list()
    alphas = list()
    for iteration in count(1):
        print 79*'*'
        print 'Iteration %d...' % iteration
        print 79*'*'
        print
        print 'Weighted Examples:'
        print_dataset(data, weights, labels)
        print

        print 'Training Classifier...'
        classifier = NaiveBayesClassifier(data, weights, labels)
        print

        print 'Predictions:'
        predictions = map(classifier.classify, data)
        print_dataset(data, weights, labels, predictions)
        print

        epsilon = sum(w for w, l, p in zip(weights, labels, predictions)
                        if l != p)
        print 'epsilon_%d: %.5f' % (iteration, epsilon)
        if epsilon == 0:
            print 'Breaking since epsilon_%d = 0' % iteration
            break

        alpha = 0.5*log((1 - epsilon) / epsilon)
        print 'alpha_%d: %.5f' % (iteration, alpha)
        print

        classifiers.append(classifier)
        alphas.append(alpha)

        if epsilon > 0.5:
            print 'Breaking since epsilon_%d > 0.5' % iteration
            break

        print 'Updating Weights...'
        factors = [(2*l - 1)*(2*p - 1) for l, p in zip(labels, predictions)]
        weights = [w*exp(-alpha*f) for w, f in zip(weights, factors)]
        Z = sum(weights)
        weights = [w/Z for w in weights]

    return AdaboostClassifier(classifiers, weights)

def print_dataset(data, weights, labels, predictions=None):
    assert len(data[0]) == 2
    if predictions is None:
        print '+-------+--------+-------+'
        print '|  w_i  |  data  | label |'
        print '+-------+--------+-------+'
        for row, w, l in zip(data, weights, labels):
            datastr = '  '.join(map(str, row))
            print  '| %.3f |  %s  |%s|' % (w, datastr, str(l).center(7))
        print '+-------+--------+-------+'
    else:
        print '+-------+--------+-------+------------+'
        print '|  w_i  |  data  | label | prediction |'
        print '+-------+--------+-------+------------+'
        for row, w, l, p in zip(data, weights, labels, predictions):
            datastr = '  '.join(map(str, row))
            line = ('| %.3f |  %s  |%s|%s|' %
                    (w, datastr, str(l).center(7), str(p).center(12)))
            print line
        print '+-------+--------+-------+------------+'

def main(step):
    ab = adaboost(DATA, LABELS)
    print ab.classify((2, 0))

if __name__ == '__main__':
    from sys import argv
    main('nostep' not in argv)
