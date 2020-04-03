import numpy as np


def generate_gaussian_distributed_samples(mus, sigma, ps, n):
    """
    Generate gaussian distributed samples from specified expectation and covariance matrix.

    Parameters
    ----------
    mus : array_like
        Expectations of gaussian distributions.
    sigma : array_like
       Covariance matrix of gaussian distributions.
    ps:
        Sampling probability of each class.
    n : int
        number of samples

    Returns
    -------
    samples :
        samples compiled to gaussian distribution
    labels :
        Sample labels.

    """

    mus, sigma, ps = np.asarray(mus), np.asarray(sigma), np.asarray(ps)
    ns = np.around(ps * n).astype(int)
    samples, labels = np.zeros((n, mus.shape[1])), np.zeros(n)
    for i in range(ns.size):
        mu, ni = mus[i, :], ns[i]
        beg, end = sum(ns[:i]), sum(ns[:i + 1])
        samples[beg:end, :] = np.random.multivariate_normal(mu, sigma, (ni,))
        labels[beg:end] = [i] * (end - beg)
    index = np.arange(n)
    np.random.shuffle(index)
    samples, labels = samples[index, :], labels[index]
    labels.shape = (n, 1)
    return np.hstack((samples, labels))


def create_data_set(mus=None, ps=None, sigma=None, n=800):
    """
    Create data set for classification and output to txt file
    """
    if mus is None:
        mus = [[1, 0], [0, -1], [0, 1]]
    if ps is None:
        ps = [.3, .6, .1]
    if sigma is None:
        sigma = [[.1, 0], [0, .1]]

    samples = generate_gaussian_distributed_samples(mus, sigma, ps, n)
    np.savetxt('*.data', samples, fmt='%.4f')


def load_samples(filename=None):
    """
    load samples and labels from txt file
    """
    if filename is None:
        filename = '*.data'
    samples = np.loadtxt(filename)
    n = samples.shape[0]
    samples, labels = samples[:, 0:-1], samples[:, -1].astype(int).reshape((n, 1))
    return samples, labels


class LinearDiscriminativeModel:
    """
    Linear discriminative model for classification with softmax regression and argmax classification strategy
    """

    def __init__(self):
        self.n_classes = 0
        self.w = np.zeros((0,))

    def train(self, samples, labels, learning_rate=0.9, max_epochs=50, mini_batch_size=128):
        """
        Train the model using samples and labels

        Parameters
        ----------
        samples : array_like
            Training samples in rows.
        labels : array_like
            Sample labels from 0 to n-1.
        learning_rate : float, optional
            Learning rate of softmax regression.
        max_epochs : int, optional
            Max number of training epochs.
        mini_batch_size: int, optional
            Mini batch size of training.

        Examples
        --------
        >>> model = LinearDiscriminativeModel()
        >>> model.train(samples, labels, max_epochs=100)

        """
        xs = np.asmatrix(samples)
        n = xs.shape[0]
        self.n_classes = int(labels.max() + 1)
        ys = np.asmatrix(np.zeros((self.n_classes, n)))
        alpha, iter_times = learning_rate, int(np.ceil(n / mini_batch_size))
        for i in range(self.n_classes):
            idx = (labels == i).reshape((labels.size,))
            ys[i, idx] = 1
        self.w = np.asmatrix(np.zeros((xs.shape[1], self.n_classes)))

        print('Epoch\tAccuracy')
        for epoch in range(max_epochs):
            for i in range(iter_times):
                beg = mini_batch_size * i
                if i == iter_times - 1:
                    end = n
                else:
                    end = mini_batch_size * (i + 1) + 1
                x, y = xs[beg: end, :].T, ys[:, beg: end].T
                wx = self.w.T.dot(x).T
                y1 = softmax(wx)
                delta = np.asmatrix(np.zeros(self.w.shape))
                for j in range(x.shape[1]):
                    delta = delta + x[:, j].dot(y[j, :] - y1[j, :])
                delta = delta / x.shape[1]
                self.w = self.w + alpha * delta
            labels1 = self.predict(samples)
            acc = sum(labels == labels1) / labels.size
            print('{epoch}\t\t{acc}'.format_map({"epoch": epoch + 1, "acc": acc[0, 0]}))

    def predict(self, samples):
        """
        Predict the labels of input samples

        Parameters
        ----------
        samples : array_like
            Samples in rows.

        Returns
        -------
        labels : list of int
            Sample labels from 0 to n-1.

        Examples
        --------
        >>> model = LinearDiscriminativeModel()
        >>> model.train(training_samples, training_labels)
        >>> pred = model.predict(testing_samples)
        >>> acc = sum(pred == testing_labels) / testing_labels.size

        """
        xs = np.asmatrix(samples)
        wx = self.w.T.dot(xs.T).T
        ys = softmax(wx)
        labels = np.argmax(ys, axis=1)
        return labels


class LinearGenerativeModel:
    """
    Linear generative model for classification using gaussian distributions
    """
    def __init__(self):
        self.n_classes = 2
        self.w = np.zeros((0,))
        self.b = np.zeros((0,))
        self.sigma = np.zeros((0,))

    def train(self, samples, labels):
        """
        Train the model using samples and labels

        Parameters
        ----------
        samples : array_like
            Training samples in rows.
        labels : array_like
            Sample labels from 0 to n-1.

        Examples
        --------
        >>> model = LinearGenerativeModel()
        >>> model.train(samples, labels)

        """
        xs = np.asmatrix(samples)
        self.n_classes = int(labels.max() + 1)
        mus = np.asmatrix(np.zeros((xs.shape[1], self.n_classes)))
        self.w = np.asmatrix(np.zeros((xs.shape[1], self.n_classes)))
        self.b = np.asmatrix(np.zeros((self.n_classes, 1)))
        self.sigma = np.asmatrix(np.zeros((xs.shape[1], xs.shape[1])))
        ps = np.asmatrix([sum(labels == i) / xs.shape[0] for i in range(self.n_classes)])
        for i in range(self.n_classes):
            idx = (labels == i).reshape((labels.size,))
            mus[:, i] = np.mean(xs[idx, :], axis=0).T
            self.sigma = self.sigma + ps[i, 0] * np.cov(samples[idx, :].T)
        sigma_inv = self.sigma.I
        for i in range(self.n_classes):
            self.w[:, i] = sigma_inv.dot(mus[:, i])
            self.b[i, 0] = -1 / 2 * mus[:, i].T.dot(self.w[:, i]) + np.log(ps[i, 0])

    def predict(self, samples):
        """
        Predict the labels of input samples

        Parameters
        ----------
        samples : array_like
            Samples in rows.

        Returns
        ----------
        labels : list of int
            Sample labels from 0 to n-1.

        Examples
        ----------
        >>> model = LinearGenerativeModel()
        >>> model.train(training_samples, training_labels)
        >>> pred = model.predict(testing_samples)
        >>> acc = sum(pred == testing_labels) / testing_labels.size

        """
        a = (self.w.T.dot(samples.T) + self.b).T
        ys = softmax(a)
        labels = np.argmax(ys, axis=1)
        return labels


def softmax(x):
    x = np.asmatrix(x)
    x = x - np.max(x, axis=1)
    return np.exp(x) / np.sum(np.exp(x), axis=1)


def main():
    create_data_set()
    samples, labels = load_samples('*.data')
    model1 = LinearDiscriminativeModel()
    model1.train(samples, labels, max_epochs=50)
    labels1 = model1.predict(samples)
    acc = sum(labels == labels1) / labels.size
    print(model1.w)
    print(acc)
    model2 = LinearGenerativeModel()
    model2.train(samples, labels)
    labels2 = model2.predict(samples)
    acc = sum(labels == labels2) / labels.size
    print(model2.w)
    print(acc)


if __name__ == '__main__':
    main()
