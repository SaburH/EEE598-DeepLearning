import numpy as np
from scipy import stats
from scipy.spatial import distance


class KNN(object):
    def __init__(self, k=3):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x, y):
        """
        Fit the model to the data

        For K-Nearest neighbors, the model is the data, so we just
        need to store the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        self.x_train=np.array(x)
        self.y_train=np.array(y)

        # raise NotImplementedError

    def predict(self, x):
        """
        Predict x from the k-nearest neighbors

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A vector of size N of the predicted class for each sample in x
        """
        # raise NotImplementedError
        dst = list()
        neighbors = np.zeros(len(x))
        # temp = np.zeros(len(x_test.shape))
        counts = np.zeros(len(x))
        for i in range(len(x)):
            temp = np.zeros(len(x))
            for j in range(len(self.x_train)):
                t = distance.euclidean(x[i, :], self.x_train[j, :])
                temp[j] = t
            sorted = np.argsort(temp)
            labelList = self.y_train[sorted[0:self.k]]
            # neighbors.append(self.y_train[sorted[i]])
            neighbors[i], counts[i] = stats.mode(labelList, axis=None)
        return neighbors




  ############Mohammad#############3#
        # dim = x.shape
        # if len(dim) == 1: # Insure that x is in 2D format
        #     x = np.reshape(x, (1,dim[0]))
        #     dim = x.shape
        # labels = np.zeros(dim[0])
        # counts = np.zeros(dim[0])
        # for i in range(0,dim[0]): # Looping over the test inputs
        #     sample = x[i,:]
        #     diff = sample - self.x_train
        #     sqDiff = np.power(diff, 2)
        #     sumSqDiff = np.sum(sqDiff, axis=1)
        #     dist = np.sqrt(sumSqDiff)
        #     nearNeigh = np.argsort(dist) # The indices of the nearest neighbours
        #     labelList = self.y_train[nearNeigh[0:self.k]]
        #     labels[i],counts[i] = stats.mode(labelList, axis=None)
        # return labels


KNN()
