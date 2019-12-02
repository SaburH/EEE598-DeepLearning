import numpy as np
import math

class FullLayer(object):
    def __init__(self, n_i, n_o):
        """
        Fully connected layer

        Parameters
        ----------
        n_i : integer
            The number of inputs
        n_o : integer
            The number of outputs
        """
        self.x = None
        self.W_grad = None
        self.b_grad = None
        new_n_i=float(n_i)
        new_n_o=float(n_o)
        var = np.sqrt(2/(new_n_i+new_n_o))
        self.W = var*np.random.randn(n_o,n_i)
        # self.W= np.random.normal(0,math.sqrt(2.0/float(n_i+n_o)),(n_o,n_i))
        self.b= np.zeros(n_o)

        # need to initialize self.W and self.b
        # raise NotImplementedError

    def forward(self, x):
        """
        Compute "forward" computation of fully connected layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        # raise NotImplementedError
        self.x=x
        out=np.dot(x,self.W.T) +self.b
        return out


    def backward(self, y_grad):
        """
        Compute "backward" computation of fully connected layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.W_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        # raise NotImplementedError
        dl_x_dir = np.dot(y_grad,self.W)
        self.W_grad = np.dot (y_grad.T, self.x)
        temp_b_grad = np.sum(y_grad, axis=0)
        self.b_grad = np.reshape(temp_b_grad, self.b.shape)
        return dl_x_dir

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        # raise NotImplementedError
        self.W= self.W - (lr*self.W_grad)
        self.b = self.b- (lr * self.b_grad)
