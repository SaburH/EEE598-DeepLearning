import numpy as np


class MaxPoolLayer(object):
    def __init__(self, size=2):
        """
        MaxPool layer
        Ok to assume non-overlapping regions
        """
        self.locs = None  # to store max locations
        self.size = size  # size of the pooling

    def forward(self, x):
        """
        Compute "forward" computation of max pooling layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the maxpooling

        Stores
        -------
        self.locs : np.array
             The locations of the maxes (needed for back propagation)
        """
        k = self.size
        y = np.zeros( (x.shape[0], x.shape[1], int( np.floor(x.shape[2]/2) ), int( np.floor(x.shape[3]/2.0) )) )
        mask = np.zeros(x.shape)
        col_max = int( np.floor( x.shape[3]/k ) )* k
        row_max = int( np.floor( x.shape[2]/k ) )* k
        for sample in range(x.shape[0]):
            for feat in range(x.shape[1]):
                row_ind = 0
                for row in range(0,row_max , k):
                    col_ind = 0
                    for col in range(0, col_max, k):
                        y[sample,feat,row_ind,col_ind] = np.amax(x[sample,feat,row:row+k,col:col+k],axis=(0,1))
                        mask[sample,feat,row:row+k,col:col+k] = x[sample,feat,row:row+k,col:col+k] == y[sample,feat,row_ind,col_ind]
                        col_ind += 1
                    row_ind += 1
        self.locs = mask
        return y
        # raise NotImplementedError

    def backward(self, y_grad):
        """
        Compute "backward" computation of maxpool layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        dLdx = np.zeros(self.locs.shape)
        col_max = int(np.floor(self.locs.shape[3] / self.size)) * self.size
        row_max = int(np.floor(self.locs.shape[2] / self.size))* self.size
        for sample in range(self.locs.shape[0]):
            for feat in range(self.locs.shape[1]):
                row_ind = 0
                for row in range(0, row_max, self.size):

                    col_ind = 0
                    for col in range(0, col_max, self.size):

                        wind = self.locs[sample,feat,row:row+self.size,col:col+self.size]
                        dLdx[sample,feat,row:row+self.size,col:col+self.size] = wind*y_grad[sample,feat,row_ind,col_ind]
                        col_ind += 1
                    row_ind += 1

        return dLdx
        # raise NotImplementedError

    def update_param(self, lr):
        pass
