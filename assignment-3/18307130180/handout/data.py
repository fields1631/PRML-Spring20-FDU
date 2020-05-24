import pickle
import numpy as np
from numpy.random import multivariate_normal, shuffle, random
from scipy.special import softmax


def createDataset(parser):
    means, covs, coeffs = initializeGaussianDistributions(parser)
    n = parser.numOfSamples
    counts = coeffs * n
    counts = counts.astype(int)
    counts[0] = counts[0] + (n - np.sum(counts))
    dataset = np.zeros([parser.numOfSamples, parser.dimOfDistributions])

    beg, end = 0, 0
    for i in range(parser.numOfDistributions):
        mean, cov = means[i], covs[i]
        beg, end = end, end + counts[i]
        dataset[beg:end] = multivariate_normal(mean, cov, counts[i])

    indices = np.arange(n)
    shuffle(indices)
    dataset = dataset[indices]
    pickle.dump(dataset, open(parser.datasetPath, 'wb'))
    if parser.saveLabels:
        labels = {'means': means, 'covs': covs, 'labels': indices}
        pickle.dump(labels, open(parser.labelsPath, 'wb'))

    return dataset


def initializeGaussianDistributions(parser):
    def isPositiveDefinite(x):
        return np.all(np.linalg.eigvals(x) > 0)

    num, dim = parser.numOfDistributions, parser.dimOfDistributions
    means = random([num, dim]) * parser.separateDegree
    covs = np.zeros([num, dim, dim])
    i = 0
    while i < parser.numOfDistributions:
        x = random([dim, dim])
        x = x + x.transpose()
        if isPositiveDefinite(x):
            covs[i] = x
            i = i + 1

    coeffs = softmax(random(num))
    return means, covs, coeffs
