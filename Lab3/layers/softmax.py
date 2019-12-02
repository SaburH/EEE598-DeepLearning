import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax

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
        self.y : np.array
             The output of the layer (needed for backpropagation)
        """
        # raise NotImplementedError
        # max = np.amax(x, axis=1)
        # newmax = np.reshape(max, (x.shape[0], 1))
        # diff = np.tile(newmax, (1, x.shape[1]))
        # y = x - diff
        # exp=np.exp(y)
        # sum=np.sum(exp,axis=1)
        # sumA=np.tile(sum,(1,x.shape[1]))
        # sumA=np.reshape(sumA,x.shape)
        # output=exp/sumA
        # self.y =output
        # return output

        maxV = np.amax(x, axis=1)
        newmax = np.reshape(maxV, (x.shape[0], 1))
        diff = np.tile(newmax, (1, x.shape[1]))
        y = x - diff
        UpPart = np.exp(y)
        LWPart = np.sum(UpPart,axis=1)
        LWPart = np.reshape(LWPart, (y.shape[0],1))
        temp = np.tile(LWPart,(1,y.shape[1]))
        # sumA=np.reshape(sumA,x.shape)
        output=UpPart/temp
        self.y = output
        return output

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

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
        # len=self.y.shape
        # output=np.zeros(y_grad.shape)
        # for i in range( len[0]):
        #     z=self.y[i,:]
        #     diag=np.diag(z)
        #     # z = np.reshape(z, (1, z.shape[0]))
        #     # mult=np.dot(z.T,z)
        #     mult = np.outer(z,z)
        #     jaq=diag - mult
        #     y_temp=np.reshape(y_grad[i,:],(1,y_grad.shape[1]))
        #     output[i,:]=np.dot(y_temp,jaq)
        # return output
        len=self.y.shape
        output=np.zeros(y_grad.shape)
        for i in range( len[0]):
            z=self.y[i,:]
            diag=np.diag(z)

            mult = np.outer(z,z)
            jaq=diag - mult
            y_temp=np.reshape(y_grad[i,:],(1,y_grad.shape[1]))
            output[i,:]=np.dot(y_temp,jaq)
        return output


    def update_param(self, lr):
        pass  # no learning for softmax layer
