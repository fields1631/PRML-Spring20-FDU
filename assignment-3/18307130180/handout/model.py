import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from scipy.stats import multivariate_normal, norm

from .data import initializeGaussianDistributions


class gaussianMixtureModel:
    def __init__(self, parser):
        super().__init__()
        self.numOfDistributions = parser.numOfDistributions
        self.dimOfDistributions = parser.dimOfDistributions
        self.means, self.covs, self.coeffs = initializeGaussianDistributions(parser)
        cnames = list(colors.cnames.keys())
        indices = np.arange(len(cnames))
        np.random.shuffle(indices)
        self.colors = [cnames[index] for index in indices[0: self.numOfDistributions]]

    def __call__(self, dataset):
        probabilities = np.zeros([dataset.shape[0], self.numOfDistributions])
        for i in range(self.numOfDistributions):
            mean, cov = self.means[i], self.covs[i]
            probabilities[:, i] = multivariate_normal.pdf(dataset, mean, cov)

        return probabilities

    def fit(self, parser, dataset):
        if parser.plot:
            fig = plt.figure()
            plt.ion()

        if parser.epsilonSpecified:
            epsilon = parser.epsilon
            probabilities = self.__call__(dataset)
            delta = 1
            logP = np.sum(np.log(np.matmul(self.coeffs, probabilities.transpose())))
            while delta > epsilon:
                if parser.plot:
                    self.plot(dataset)
                    plt.pause(0.2)
                self.expectationMaximumStep(dataset)
                logP1 = np.sum(np.log(np.matmul(self.coeffs, probabilities.transpose())))
                delta = logP1 - logP
                logP = logP1
        elif parser.epochsSpecified:
            epochs = parser.epochs
            for epoch in range(epochs):
                print(epoch)
                if parser.plot:
                    self.plot(dataset)
                    # plt.pause(0.2)
                self.expectationMaximumStep(dataset)
        else:
            raise ValueError('Training method must be specified in command line options.')
        if parser.plot:
            plt.ioff()
            plt.show()

    def expectationMaximumStep(self, dataset):
        N = dataset.shape[0]
        gamma = np.zeros([N, self.numOfDistributions])
        for i in range(self.numOfDistributions):
            mean, cov = self.means[i], self.covs[i]
            gamma[:, i] = multivariate_normal.pdf(dataset, mean, cov)

        gamma = (gamma.transpose() / np.sum(gamma, axis=1)).transpose()
        Nk = np.sum(gamma, axis=0)
        means = self.means
        self.coeffs = Nk / N
        self.means = np.matmul(gamma.transpose(), dataset)
        self.means = (self.means.transpose() / Nk).transpose()
        for i in range(self.numOfDistributions):
            delta = dataset - means[i]
            self.covs[i] = np.matmul(delta.transpose(), delta) / Nk[i]

    def plot(self, dataset):
        probabilities = self.__call__(dataset)
        labels = np.argmax(probabilities, axis=1)
        indices = []
        for i in range(self.numOfDistributions):
            [index] = np.where(labels == i)
            indices.append(index)

        plt.cla()
        if dataset.shape[1] > 1:
            dataset = dataset[:, 0:2]
            means, covs = self.means[:, 0:2], self.covs[:, 0:2, 0:2]
            for i in range(self.numOfDistributions):
                index = indices[i]
                if index is not None:
                    mean, cov = means[i], covs[i]
                    vals, vecs = np.linalg.eigh(cov)
                    order = vals.argsort()[::-1]
                    vals, vecs = vals[order], vecs[:, order]
                    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    width, height = 2 * np.sqrt(vals)
                    ellipse = patches.Ellipse(xy=mean, width=width, height=height, angle=theta, fill=False, color=self.colors[i])
                    plt.scatter(dataset[index, 0], dataset[index, 1], c=self.colors[i])
                    ax = plt.gca()
                    ax.add_artist(ellipse)

            plt.axis('auto')
            plt.show()
        else:
            means, covs = self.means, self.covs
            for i in range(self.numOfDistributions):
                index = indices[i]
                mean, cov = means[i], covs[i]
                std = np.sqrt(cov)
                xs = np.linspace(mean - 3 * std, mean + 3 * std, 50)
                ys = norm.pdf((xs - mean) / std)
                plt.scatter(dataset[index], np.zeros_like(dataset[index]), c=self.colors[i])
                plt.plot(xs, ys, c=self.colors[i])


def trainModel(parser):
    dataset = pickle.load(open(parser.datasetPath, 'rb'))
    if parser.resume:
        model = pickle.load(open(parser.modelPath, 'rb'))
    else:
        model = gaussianMixtureModel(parser)

    model.fit(parser, dataset)
    return model


def evaluate(parser):
    labels = pickle.load(open(parser.labelsPath, 'rb'))
