import numpy as np

class FlattenLayer(object):
    def __init__(self):
        """
        Flatten layer
        """
        self.orig_shape = None # to store the shape for backpropagation

    def forward(self, x):
        """
        Compute "forward" computation of flatten layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the flatten operation
            size = training samples x (number of input channels * number of rows * number of columns)
            (should make a copy of the data with np.copy)

        Stores
        -------
        self.orig_shape : list
             The original shape of the data
        """
        # raise NotImplementedError
        size= (x.shape[0],  x.shape[1], x.shape[2], x.shape[3])
        mult = x.shape[1]* x.shape[2]* x.shape[3]
        self.orig_shape = size
        y=np.copy(x)
        y=np.reshape(y,(x.shape[0],mult))
        return y
    def backward(self, y_grad):
        """
        Compute "backward" computation of flatten layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        # raise NotImplementedError
        new_grad= np.copy(y_grad)
        new_grad = np.reshape(new_grad,(self.orig_shape))
        return new_grad
    def update_param(self, lr):
        pass
